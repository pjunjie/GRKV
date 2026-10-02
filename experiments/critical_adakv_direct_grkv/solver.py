# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Total-anchor ridge with BF16 objective checks and explicit bounded approximations."""

from dataclasses import asdict, dataclass

import torch

from experiments.critical_adakv_behavior_v2.cache import Block
from experiments.critical_adakv_postproj_grkv.operators import AttentionMap, ResidualMetric, ridge_update
from experiments.critical_adakv_postproj_grkv.stable_solver import promote_observations


@dataclass(frozen=True)
class FitConfig:
    objective: str = "pre"
    steps: int = 1
    ridge: float = 0.01
    relative_cap: float = 0.1
    max_iterations: int = 16
    tolerance: float = 1e-6
    precision: str = "float32"

    def __post_init__(self):
        if self.objective not in ("pre", "diag", "mix") or self.steps not in (1, 2):
            raise ValueError("Unsupported direct objective/steps")
        if self.ridge <= 0 or self.relative_cap <= 0 or self.max_iterations <= 0 or self.tolerance <= 0:
            raise ValueError("Positive ridge, norm cap, and iteration controls required")
        if self.precision not in ("float32", "float64"):
            raise ValueError("Unsupported precision")


class DiagonalMetric:
    def __init__(self, diagonal, weights):
        self.root = weights.sqrt()[:, None] * diagonal.sqrt()[None, :]

    def forward(self, residual):
        return residual * self.root

    def adjoint(self, residual):
        return residual * self.root


def value_dual(op, keys, values, target, weights, diagonal, ridge):
    """Independent H,D small systems; exact pre/diagonal metric ridge update."""
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
    # Each GQA output channel can have a different diagonal weight. Fold it into
    # A and r per value channel, retaining sharing across channels for pre loss.
    if diagonal is None:
        matrix = gram + ridge * torch.eye(groups * queries, device=a.device, dtype=a.dtype)
        dual = torch.linalg.solve(matrix, r)
        delta = a.transpose(-1, -2) @ dual
        relative = torch.linalg.vector_norm(matrix @ dual - r) / torch.linalg.vector_norm(r).clamp_min(1e-30)
    else:
        d = diagonal.reshape(heads, groups, dim).permute(0, 2, 1)
        d = d.repeat_interleave(queries, -1).sqrt()
        matrix = gram[:, None] * d[..., :, None] * d[..., None, :]
        matrix = matrix + ridge * torch.eye(groups * queries, device=a.device, dtype=a.dtype)
        rhs = r.transpose(-1, -2) * d
        dual = torch.linalg.solve(matrix, rhs[..., None])[..., 0]
        delta = torch.einsum("hrs,hdr->hsd", a, dual * d)
        relative = torch.linalg.vector_norm((matrix @ dual[..., None])[..., 0] - rhs)
        relative = relative / torch.linalg.vector_norm(rhs).clamp_min(1e-30)
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
def fit_direct(anchor, observations, projection, config=FitConfig(), diagonal=None):
    dtype = getattr(torch, config.precision)
    obs = promote_observations(observations, anchor.key.device, dtype)
    k0, v0 = anchor.key[0].to(dtype), anchor.value[0].to(dtype)
    op = AttentionMap(obs, anchor.positions[0], anchor.valid[0], anchor.regression[0])
    metric: DiagonalMetric | ResidualMetric
    if config.objective == "pre":
        metric = DiagonalMetric(torch.ones(obs.target.shape[-1], device=k0.device, dtype=dtype), obs.weights)
    elif config.objective == "diag":
        diagonal = diagonal.to(dtype)
        metric = DiagonalMetric(diagonal, obs.weights)
    else:
        metric = ResidualMetric(projection.to(dtype), obs.weights, True)

    def objective(k, v):
        e = metric.forward(op.state(k, v)[0] - obs.target)
        return float(e.square().sum() + config.ridge * ((k - k0).square().sum() + (v - v0).square().sum()))

    def safe(k, v):
        if not bool(torch.isfinite(k).all() and torch.isfinite(v).all()):
            return float("inf")
        return objective(k, v)

    k, v = k0, v0
    initial = objective(k0, v0)
    records = []
    for step in range(config.steps):
        # At fixed K, solve for TOTAL V-V0, including round 2.
        vf, va = op.derivatives(k, v0, "value")
        if config.objective == "mix":
            dv, vr = ridge_update(
                vf,
                va,
                metric,
                obs.target - op.state(k, v0)[0],
                config.ridge,
                config.max_iterations,
                config.tolerance,
            )
        else:
            dv, vr = value_dual(op, k, v0, obs.target, obs.weights, diagonal, config.ridge)
        v1 = round_bounded(v0, dv, op.mutable, config.relative_cap, anchor.value.dtype)
        kf, ka = op.derivatives(k, v1, "key")
        # y(K) + J * (K0 + total_delta - K); regularize total_delta.
        rhs = obs.target - op.state(k, v1)[0] + kf(k - k0)
        dk, kr = ridge_update(
            kf,
            ka,
            metric,
            rhs,
            config.ridge,
            config.max_iterations,
            config.tolerance,
        )
        k1 = round_bounded(k0, dk, op.mutable, config.relative_cap, anchor.key.dtype)
        states = [(k0, v0), (k, v), (k, v1), (k1, v1)]
        objectives = [safe(*state) for state in states]
        selected = min(range(len(states)), key=objectives.__getitem__)
        k, v = states[selected]
        records.append(
            dict(
                step=step + 1,
                value_solver=vr,
                key_solver=kr,
                objectives=[x if x < float("inf") else None for x in objectives],
                selected=("baseline", "previous", "value", "key_value")[selected],
            )
        )
    fixed = ~op.mutable
    if not torch.equal(k[fixed], k0[fixed]) or not torch.equal(v[fixed], v0[fixed]):
        raise RuntimeError("Direct fit changed a protected slot")
    result = Block(
        k[None].to(anchor.key.dtype), v[None].to(anchor.value.dtype), anchor.positions, anchor.valid, anchor.regression
    )
    return result, dict(
        config=asdict(config),
        initial_objective=initial,
        final_objective=objective(k, v),
        rounds=records,
        total_query_weight=float(obs.weights.sum()),
        query_count=len(obs.weights),
        anchor="CriticalAdaKV",
        total_anchor_regularization=True,
    )
