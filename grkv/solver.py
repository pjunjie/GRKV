# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRKV Default: one regression round with independent key and value caps."""

from dataclasses import asdict

import torch

from grkv.cache import Block
from grkv.io import tensor_sha
from grkv.operators import AttentionMap, promote_observations, ridge_update


class DiagonalMetric:
    def __init__(self, diagonal, weights):
        self.root = weights.sqrt()[:, None] * diagonal.sqrt()[None, :]

    def forward(self, residual):
        return residual * self.root

    def adjoint(self, residual):
        return residual * self.root


def value_dual(op, keys, values, target, weights, ridge):
    """Independent H,D small systems; exact pre-projection ridge update."""
    _, attention, _ = op.state(keys, values)
    heads, groups, queries, slots = attention.shape
    dim = values.shape[-1]
    a = (attention * op.mutable[:, None, None, :]).reshape(heads, groups * queries, slots)
    root_w = weights.sqrt().repeat(groups)
    a = a * root_w[None, :, None]
    residual = target - op.state(keys, values)[0]
    r = residual.reshape(queries, heads, groups, dim).permute(1, 2, 0, 3).reshape(heads, -1, dim)
    r = r * root_w[None, :, None]
    gram = a @ a.transpose(-1, -2)
    matrix = gram + ridge * torch.eye(groups * queries, device=a.device, dtype=a.dtype)
    dual = torch.linalg.solve(matrix, r)
    delta = a.transpose(-1, -2) @ dual
    relative = torch.linalg.vector_norm(matrix @ dual - r) / torch.linalg.vector_norm(r).clamp_min(1e-30)
    return delta, dict(method="batched_value_dual", relative_residual=float(relative), iterations=1)


def round_bounded(base, delta, mutable, cap, storage_dtype):
    """Per-KV-head total K/V correction bound, rechecked after BF16 writeback."""
    mask = mutable[..., None]
    delta = torch.where(mask, delta, 0)
    norm = torch.linalg.vector_norm(base * mask, dim=(-2, -1), keepdim=True)
    limit = cap * norm
    scale = (limit / torch.linalg.vector_norm(delta, dim=(-2, -1), keepdim=True).clamp_min(1e-30)).clamp(max=1)
    out = torch.where(mask, (base + scale * delta).to(storage_dtype).to(base.dtype), base)
    # Quantization can push a boundary solution outside its cap. Shrink only
    # affected heads, then fall back exactly if the cap cannot be met.
    for _ in range(4):
        over = torch.linalg.vector_norm(out - base, dim=(-2, -1), keepdim=True) > limit
        if not bool(over.any()):
            break
        scale = torch.where(over, scale * 0.9, scale)
        out = torch.where(mask, (base + scale * delta).to(storage_dtype).to(base.dtype), base)
    over = torch.linalg.vector_norm(out - base, dim=(-2, -1), keepdim=True) > limit
    return torch.where(over, base, out)


@torch.no_grad()
def fit_default(anchor, observations, projection, config):
    dtype = getattr(torch, config.precision)
    obs = promote_observations(observations, anchor.key.device, dtype)
    k0, v0 = anchor.key[0].to(dtype), anchor.value[0].to(dtype)
    op = AttentionMap(obs, anchor.positions[0], anchor.valid[0], anchor.regression[0])
    metric = DiagonalMetric(torch.ones(obs.target.shape[-1], device=k0.device, dtype=dtype), obs.weights)

    def objective(k, v):
        residual = metric.forward(op.state(k, v)[0] - obs.target)
        return float(residual.square().sum() + config.ridge * ((k - k0).square().sum() + (v - v0).square().sum()))

    def safe(k, v):
        if not bool(torch.isfinite(k).all() and torch.isfinite(v).all()):
            return float("inf")
        return objective(k, v)

    initial = objective(k0, v0)
    dv, vr = value_dual(op, k0, v0, obs.target, obs.weights, config.ridge)
    v1 = round_bounded(v0, dv, op.mutable, config.relative_cap, anchor.value.dtype)
    kf, ka = op.derivatives(k0, v1, "key")
    rhs = obs.target - op.state(k0, v1)[0] + kf(k0 - k0)
    dk, kr = ridge_update(kf, ka, metric, rhs, config.ridge, config.max_iterations, config.tolerance)
    updated_keys = round_bounded(k0, dk, op.mutable, config.key_relative_cap, anchor.key.dtype)
    states = [(k0, v0), (k0, v0), (k0, v1), (updated_keys, v1)]
    objectives = [safe(*state) for state in states]
    selected = min(range(len(states)), key=objectives.__getitem__)
    k, v = states[selected]
    fixed = ~op.mutable
    assert torch.equal(k[fixed], k0[fixed]) and torch.equal(v[fixed], v0[fixed])
    result = Block(
        k[None].to(anchor.key.dtype),
        v[None].to(anchor.value.dtype),
        anchor.positions,
        anchor.valid,
        anchor.regression,
    )
    record = dict(
        config=asdict(config),
        initial_objective=initial,
        final_objective=objective(k, v),
        rounds=[
            dict(
                step=1,
                value_solver=vr,
                key_solver=kr,
                objectives=[x if x < float("inf") else None for x in objectives],
                selected=("baseline", "previous", "value", "key_value")[selected],
            )
        ],
        total_query_weight=float(obs.weights.sum()),
        query_count=len(obs.weights),
        anchor="CriticalAdaKV",
        total_anchor_regularization=True,
        independent_key_cap=config.key_relative_cap,
    )
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
