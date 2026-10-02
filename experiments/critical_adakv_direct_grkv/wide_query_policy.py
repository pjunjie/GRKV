# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Wide context Q sampling with unchanged selected student KV and protected window."""

from dataclasses import dataclass, field

import torch

from experiments.critical_adakv_direct_grkv import key_policy
from experiments.critical_adakv_direct_grkv.key_policy import KeyPolicyConfig, KeyPolicyPress
from experiments.critical_adakv_grkv.press import tensor_sha

TAIL_COUNTS = {"tail32": (32,), "tail64": (32, 64), "tail128": (64, 128)}
GLOBAL_POLICIES = ("context_stratified", "head_tail", "tail_global")


@dataclass(frozen=True)
class WideQueryConfig(KeyPolicyConfig):
    query_sampling: str = "tail64"
    query_count: int = 64

    def __post_init__(self):
        super().__post_init__()
        fixed = (self.ridge, self.relative_cap, self.max_iterations, self.tolerance, self.precision)
        if fixed != (0.01, 0.1, 16, 1e-6, "float32") or self.key_relative_cap not in (0.0, 0.1):
            raise ValueError("Wide-Q protocol keeps the registered solver parameters fixed")
        if self.query_sampling == "window_q16_compat":
            if type(self.query_count) is not int or self.query_count != 16:
                raise ValueError("The compatibility control requires exactly 16 queries")
            return
        if type(self.query_count) is not int or self.query_count not in (32, 64, 128):
            raise ValueError("Wide Q count must be 32, 64, or 128")
        allowed = TAIL_COUNTS.get(self.query_sampling)
        if allowed is not None:
            if self.query_count not in allowed:
                raise ValueError("Unregistered tail range/count pair")
        elif self.query_sampling not in GLOBAL_POLICIES:
            raise ValueError("Unknown wide Q sampling policy")


def uniform(a, b, count, device):
    if not 0 < count <= b - a:
        raise ValueError("Uniform sample must fit its source interval")
    return torch.linspace(a, b - 1, count, device=device, dtype=torch.float64).round().long()


def stratified(a, b, count, device):
    if not 0 < count <= b - a:
        raise ValueError("Stratified sample must fit its source interval")
    edges = a + torch.arange(count + 1, device=device) * (b - a) // count
    return (edges[:-1] + edges[1:] - 1) // 2


def source_positions(config, context_tokens, device="cpu"):
    if type(context_tokens) is not int or context_tokens < 1:
        raise ValueError("Actual context token count must be positive")
    requested = config.query_count
    if context_tokens < requested:
        return torch.arange(context_tokens, device=device)
    policy = config.query_sampling
    if policy == "window_q16_compat":
        return uniform(max(0, context_tokens - 32), context_tokens, requested, device)
    if policy in TAIL_COUNTS:
        width = min(int(policy.removeprefix("tail")), context_tokens)
        return uniform(context_tokens - width, context_tokens, requested, device)
    if policy == "context_stratified":
        return stratified(0, context_tokens, requested, device)
    half = requested // 2
    head = (
        torch.arange(half, device=device)
        if policy == "head_tail"
        else stratified(0, context_tokens - half, half, device)
    )
    return torch.cat((head, torch.arange(context_tokens - half, context_tokens, device=device)))


def query_metadata(config, context_tokens, used_fit_q=0, skip_reason=None):
    positions = source_positions(config, context_tokens).tolist()
    effective = min(config.query_count, context_tokens)
    if used_fit_q not in (0, effective):
        raise ValueError("A fit must use exactly the planned effective Q count")
    return dict(
        requested_Q=config.query_count,
        planned_Q=effective,
        effective_Q=effective,
        used_fit_Q=used_fit_q,
        context_tokens=context_tokens,
        source_positions=positions,
        source_range=[positions[0], positions[-1] + 1],
        short_context_fallback=context_tokens < config.query_count,
        virtual_position=context_tokens - 1,
        planned_total_query_weight=6.0,
        planned_per_query_weight=6.0 / effective,
        total_query_weight=6.0 if used_fit_q else 0.0,
        skip_reason=skip_reason,
    )


@dataclass
class WideQueryPress(KeyPolicyPress):
    fit_config: WideQueryConfig = field(default_factory=WideQueryConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.window_size != 32:
            raise ValueError("Wide Q sampling never changes protected window_size=32")
        if self.mode != "none" or self.behavior_probe_windows != 9 or self.behavior_probes_per_window != 2:
            raise ValueError("Wide Q sampling is direct-only with no CV gates")

    def _source_probe_folds(self, prefix_length, device):
        positions = source_positions(self.fit_config, prefix_length + self.window_size, device)
        empty = torch.empty(0, device=device, dtype=torch.long)
        return positions, empty, empty

    @torch.no_grad()
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        tokens = keys.shape[2]
        if tokens <= self.window_size:
            # SnapKV scoring is undefined here. Preserve the entire protected input,
            # and disclose this boundary fallback rather than claiming a 90% budget.
            self._pending.pop(module.layer_idx, None)
            module.masked_key_indices = tuple(torch.empty(0, dtype=torch.long, device=keys.device) for _ in range(3))
            self.layer_records.append(
                dict(
                    layer=int(module.layer_idx),
                    **query_metadata(self.fit_config, tokens, skip_reason="context_not_larger_than_protection_window"),
                    compression_applied=False,
                    native_short_path_supported=False,
                    kept=keys.shape[1] * tokens,
                    kept_per_head=[tokens] * keys.shape[1],
                    regression_slots=0,
                    fixed=keys.shape[1] * tokens,
                    changed=0,
                    fixed_unchanged=True,
                    unchosen_unchanged=True,
                )
            )
            return keys, values
        original = key_policy.fit_key_policy

        def fit(anchor, observations, projection, config, diagonal=None):
            result, record = original(anchor, observations, projection, config, diagonal)
            mutable = anchor.regression.unsqueeze(-1).expand_as(anchor.key)
            assert torch.equal(result.key[~mutable], anchor.key[~mutable])
            assert torch.equal(result.value[~mutable], anchor.value[~mutable])
            for name in ("positions", "valid", "regression"):
                assert torch.equal(getattr(anchor, name), getattr(result, name))
            record["student_layout"] = {
                name + "_sha256": tensor_sha(getattr(anchor, name)) for name in ("positions", "valid", "regression")
            }
            record["student_layout"].update(
                slots_per_head=anchor.valid.sum(-1).flatten().tolist(),
                mutable_per_head=anchor.regression.sum(-1).flatten().tolist(),
                fixed_unchanged=True,
                student_only_selected_kv=True,
            )
            record["key_update_norm"] = float(torch.linalg.vector_norm(result.key.float() - anchor.key.float()))
            record["value_update_norm"] = float(torch.linalg.vector_norm(result.value.float() - anchor.value.float()))
            record["teacher_output_shape"] = list(observations.target.shape)
            record["query_shape"] = list(observations.queries.shape)
            return result, record

        key_policy.fit_key_policy = fit
        try:
            output = super().compress(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            key_policy.fit_key_policy = original
        record = self.layer_records[-1]
        fitted = "fit" in record
        metadata = query_metadata(
            self.fit_config,
            tokens,
            used_fit_q=min(tokens, self.fit_config.query_count) if fitted else 0,
            skip_reason=None if fitted else record.get("skip", "no_mutable_slots"),
        )
        if fitted:
            assert record["source_positions"] == metadata["source_positions"]
            assert abs(record["fit"]["total_query_weight"] - 6.0) < 1e-5
        record.update(metadata, compression_applied=True, native_short_path_supported=True)
        return output
