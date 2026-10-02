# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Private experiment adapter: unchanged CriticalAdaKV selection and GRKV solvers."""

import hashlib
from dataclasses import dataclass, field

import torch

from kvpress.presses.criticalkv_press import CriticalAdaKVPress
from kvpress.presses.defensive_grkv_press import DefensiveGRKVBehaviorCVPress, DefensiveGRKVPurePress
from kvpress.presses.snapkv_press import SnapKVPress


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass
class CaptureSnapKV(SnapKVPress):
    """Keep an unmodified score copy before CriticalAdaKV's in-place selection."""

    def score(self, *args, **kwargs):
        scores = super().score(*args, **kwargs)
        self.raw_scores = scores.clone()
        return scores


@dataclass
class CriticalAdaGRKVPress(DefensiveGRKVBehaviorCVPress):
    """Use CriticalAdaKV's exact mask with none, pure, or Behavior-CV regression.

    Inherited DefensiveGRKV methods supply the existing ragged-head GRKV
    solver only. Their DefensiveKV scoring and budget selection are overridden.
    No CriticalAdaKV or public GRKV source is modified by this experiment.
    """

    mode: str = "none"
    critical_alpha_safeguard: float = 0.2
    critical_epsilon: float = 1e-4
    critical_first_stage_ratio: float = 0.5
    layer_records: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("none", "pure", "behavior_cv", "behavior_cv_critical_fallback"):
            raise ValueError("unknown regression mode")

    def post_init_from_model(self, model):
        super().post_init_from_model(model)
        self.layer_records = []
        self.reset_behavior_cv_diagnostics()

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        if keys.shape[0] != 1:
            raise ValueError("this experiment uses the original CriticalAdaKV batch-size-one path")
        scorer = CaptureSnapKV(
            compression_ratio=self.compression_ratio,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
        )
        selector = CriticalAdaKVPress(
            scorer,
            alpha_safeguard=self.critical_alpha_safeguard,
            epsilon=self.critical_epsilon,
            first_stage_ratio=self.critical_first_stage_ratio,
        )
        selector.compress(module, hidden_states, keys, values, attentions, kwargs)
        keep = torch.ones(keys.shape[:3], device=keys.device, dtype=torch.bool)
        keep[module.masked_key_indices] = False
        self._critical_keep = keep
        self._critical_scores = scorer.raw_scores
        return scorer.raw_scores

    def _global_keep_mask(self, scores, compression_ratio):  # type: ignore[override]
        if scores is not self._critical_scores or compression_ratio != self.compression_ratio:
            raise RuntimeError("selection must follow the current CriticalAdaKV score call")
        return self._critical_keep

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        if self.mode == "none":
            with torch.no_grad():
                self.score(module, hidden_states, keys, values, attentions, kwargs)
            output_keys, output_values = keys, values
            returned_path = "critical_adakv"
        elif self.mode == "pure":
            output_keys, output_values = DefensiveGRKVPurePress.compress(
                self, module, hidden_states, keys, values, attentions, kwargs
            )
            returned_path = "pure_grkv"
        else:
            accepted_before = self.behavior_cv_diagnostics["accepted_layers"]
            output_keys, output_values = DefensiveGRKVBehaviorCVPress.compress(
                self, module, hidden_states, keys, values, attentions, kwargs
            )
            accepted = self.behavior_cv_diagnostics["accepted_layers"] > accepted_before
            returned_path = "behavior_candidate" if accepted else "pure_grkv_fallback"
            if not accepted and self.mode == "behavior_cv_critical_fallback":
                output_keys, output_values = keys, values
                returned_path = "critical_adakv_fallback"
        keep = self._critical_keep
        changed = (output_keys != keys).any(-1) | (output_values != values).any(-1)
        if bool((changed & ~keep).any()):
            raise RuntimeError("regression changed an unselected position")
        fixed, fixed_valid, regression, regression_valid = self._build_selected_layout(keep, self._critical_scores)
        fixed_changed = changed.gather(2, fixed) & fixed_valid
        if bool(fixed_changed.any()):
            raise RuntimeError("regression changed a fixed guard/window position")
        expected = keys.shape[1] * int(keys.shape[2] * (1 - self.compression_ratio))
        if int(keep.sum()) != expected:
            raise RuntimeError("CriticalAdaKV budget changed")
        if not bool(torch.isfinite(output_keys).all() and torch.isfinite(output_values).all()):
            raise RuntimeError("nonfinite cache output")
        self.layer_records.append(
            {
                "layer": int(module.layer_idx),
                "returned_path": returned_path,
                "context_tokens": keys.shape[2],
                "kept": int(keep.sum()),
                "kept_per_head": keep.sum(-1).flatten().tolist(),
                "keep_sha256": tensor_sha(keep),
                "fixed": int(fixed_valid.sum()),
                "regression_slots": int(regression_valid.sum()),
                "changed": int(changed.sum()),
                "unchosen_unchanged": True,
                "fixed_unchanged": True,
            }
        )
        return output_keys, output_values
