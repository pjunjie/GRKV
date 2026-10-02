# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Layer-local direct deployment using selected Q from the actual context prefill."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter

import torch
from transformers.models.llama.modeling_llama import rotate_half

from experiments.critical_adakv_behavior_v2.cache import Block
from experiments.critical_adakv_direct_grkv.solver import FitConfig, fit_direct
from experiments.critical_adakv_grkv.press import CriticalAdaGRKVPress, tensor_sha
from experiments.critical_adakv_postproj_grkv.operators import AttentionMap, Observations


def device_pack(keys, values, keep, layout):
    fp, fv, rp, rv = layout
    positions = torch.cat((fp, rp), -1)
    valid = torch.cat((fv, rv), -1)
    mutable = torch.cat((torch.zeros_like(fv), rv), -1)
    indices = positions[..., None].expand(-1, -1, -1, keys.shape[-1])
    return Block(
        keys.gather(2, indices) * valid[..., None],
        values.gather(2, indices) * valid[..., None],
        positions,
        valid,
        mutable,
    )


@dataclass
class DirectPress(CriticalAdaGRKVPress):
    fit_config: FitConfig = field(default_factory=FitConfig)
    audit: bool = True
    query_aware: bool = False
    reference_check: bool = False
    _pending: dict = field(default_factory=dict, init=False, repr=False)
    _diagonals: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.query_aware:
            raise ValueError("Direct fitting only accepts context prefill queries")

    def post_init_from_model(self, model):
        super().post_init_from_model(model)
        self._pending.clear()
        if model.config.model_type != "llama":
            raise ValueError("This private Q capture is qualified for Llama only")
        if self.fit_config.objective == "diag" and not self._diagonals:
            for layer in model.model.layers:
                w = layer.self_attn.o_proj.weight.detach().float()
                diagonal = w.square().sum(0)
                self._diagonals[layer.self_attn.layer_idx] = 0.5 + 0.5 * diagonal / diagonal.mean()

    @contextmanager
    def __call__(self, model):
        hooks = []

        def capture(index, module, inputs, output):
            # The hook retains only six raw Q vectors. It never runs q_proj again.
            if output.shape[1] <= self.window_size:
                return
            folds = self._source_probe_folds(output.shape[1] - self.window_size, output.device)
            if folds is not None:
                self._pending[index] = (output.index_select(1, folds[0]).detach(), folds[0])

        try:
            for layer in model.model.layers:
                index = layer.self_attn.layer_idx
                hooks.append(
                    layer.self_attn.q_proj.register_forward_hook(lambda m, i, o, index=index: capture(index, m, i, o))
                )
            with super().__call__(model):
                yield
        finally:
            for hook in hooks:
                hook.remove()
            self._pending.clear()

    @torch.no_grad()
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        begin = perf_counter()
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        keep = self._critical_keep
        layout = self._build_selected_layout(keep, scores)
        anchor = device_pack(keys, values, keep, layout)
        selected_at = perf_counter()
        captured = self._pending.pop(module.layer_idx, None)
        record = dict(
            layer=int(module.layer_idx),
            context_tokens=keys.shape[2],
            kept=int(keep.sum()),
            kept_per_head=keep.sum(-1).flatten().tolist(),
            fixed=int((anchor.valid & ~anchor.regression).sum()),
            regression_slots=int(anchor.regression.sum()),
            query_source="actual_context_prefill_q_proj",
            query_aware=False,
            sink_tokens=0,
        )
        if self.audit:
            record["keep_sha256"] = tensor_sha(keep)
        if captured is None or not bool(anchor.regression.any()):
            record["skip"] = "no_history_queries_or_mutable_slots"
            self.layer_records.append(record)
            return keys, values
        raw, sources = captured
        heads, dim = keys.shape[1], keys.shape[-1]
        q = raw.view(1, len(sources), -1, dim).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q = q * cos[:, -1:, None, :].transpose(1, 2) + rotate_half(q) * sin[:, -1:, None, :].transpose(1, 2)
        q = q[0].reshape(heads, -1, len(sources), dim).float()
        positions = torch.full_like(sources, keys.shape[2] - 1)
        obs = Observations(
            q,
            positions,
            torch.empty((len(sources), q.shape[0] * q.shape[1] * dim), device=q.device),
            torch.full(q.shape[:3], -torch.inf, device=q.device),
            torch.zeros_like(q),
            torch.full((len(sources),), 6 / len(sources), device=q.device),
        )
        full_positions = torch.arange(keys.shape[2], device=q.device).expand(heads, -1)
        full_valid = torch.ones_like(full_positions, dtype=torch.bool)
        teacher = AttentionMap(obs, full_positions, full_valid, torch.zeros_like(full_valid))
        obs.target = teacher.state(keys[0].float(), values[0].float())[0]
        del teacher
        target_at = perf_counter()
        candidate, fit_record = fit_direct(
            anchor, obs, module.o_proj.weight.detach(), self.fit_config, self._diagonals.get(module.layer_idx)
        )
        if self.reference_check and module.layer_idx == 0:
            precise, precise_record = fit_direct(
                anchor,
                obs,
                module.o_proj.weight.detach(),
                replace(self.fit_config, precision="float64", max_iterations=512),
                self._diagonals.get(module.layer_idx),
            )
            record["reference_comparison"] = dict(
                precise=precise_record,
                max_key_difference=float((candidate.key.float() - precise.key.float()).abs().max()),
                max_value_difference=float((candidate.value.float() - precise.value.float()).abs().max()),
            )
            old_q, _ = self._virtual_queries(
                module, hidden_states, kwargs["position_embeddings"], sources, keys.shape[2] - 1
            )
            record["old_recomputed_query_max_difference"] = float((old_q.reshape_as(q).float() - q).abs().max())
        for head in range(heads):
            valid = candidate.regression[0, head]
            p = candidate.positions[0, head, valid]
            keys[0, head, p] = candidate.key[0, head, valid]
            values[0, head, p] = candidate.value[0, head, valid]
        record.update(
            fit=fit_record,
            source_positions=sources.tolist(),
            virtual_position=keys.shape[2] - 1,
            selection_host_seconds=selected_at - begin,
            target_host_seconds=target_at - selected_at,
            fit_host_seconds=perf_counter() - target_at,
        )
        self.layer_records.append(record)
        return keys, values
