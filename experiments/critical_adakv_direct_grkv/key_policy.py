# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Explicit one-round pre-loss ablations with independent total K and V caps."""

from dataclasses import asdict, dataclass
from math import isfinite

import torch

from experiments.critical_adakv_behavior_v2.cache import Block
from experiments.critical_adakv_direct_grkv import press as reference_press
from experiments.critical_adakv_direct_grkv import solver as reference
from experiments.critical_adakv_postproj_grkv.operators import AttentionMap


@dataclass(frozen=True)
class KeyPolicyConfig(reference.FitConfig):
    key_relative_cap: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.objective != "pre" or self.steps != 1:
            raise ValueError("This ablation is explicitly limited to one-round pre loss")
        if not isfinite(self.key_relative_cap) or not 0 <= self.key_relative_cap <= self.relative_cap:
            raise ValueError("K cap must be finite, nonnegative and no larger than the V cap")


@torch.no_grad()
def fit_key_policy(anchor, observations, projection, config, diagonal=None):
    dtype = getattr(torch, config.precision)
    obs = reference.promote_observations(observations, anchor.key.device, dtype)
    k0, v0 = anchor.key[0].to(dtype), anchor.value[0].to(dtype)
    op = AttentionMap(obs, anchor.positions[0], anchor.valid[0], anchor.regression[0])
    metric = reference.DiagonalMetric(torch.ones(obs.target.shape[-1], device=k0.device, dtype=dtype), obs.weights)

    def objective(k, v):
        residual = metric.forward(op.state(k, v)[0] - obs.target)
        return float(residual.square().sum() + config.ridge * ((k - k0).square().sum() + (v - v0).square().sum()))

    def safe(k, v):
        if not bool(torch.isfinite(k).all() and torch.isfinite(v).all()):
            return float("inf")
        return objective(k, v)

    initial = objective(k0, v0)
    dv, vr = reference.value_dual(op, k0, v0, obs.target, obs.weights, None, config.ridge)
    v1 = reference.round_bounded(v0, dv, op.mutable, config.relative_cap, anchor.value.dtype)
    if config.key_relative_cap == 0:
        k1 = k0
        kr = dict(skipped=True, reason="explicit_value_only_ablation")
    else:
        kf, ka = op.derivatives(k0, v1, "key")
        rhs = obs.target - op.state(k0, v1)[0] + kf(k0 - k0)
        dk, kr = reference.ridge_update(kf, ka, metric, rhs, config.ridge, config.max_iterations, config.tolerance)
        k1 = reference.round_bounded(k0, dk, op.mutable, config.key_relative_cap, anchor.key.dtype)
    states = [(k0, v0), (k0, v0), (k0, v1), (k1, v1)]
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
    return result, dict(
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


@dataclass
class KeyPolicyPress(reference_press.DirectPress):
    """Preserve the proven context-Q capture and only replace the local fit call."""

    @torch.no_grad()
    def compress(self, *args, **kwargs):
        original = reference_press.fit_direct
        setattr(reference_press, "fit_direct", fit_key_policy)
        try:
            return super().compress(*args, **kwargs)
        finally:
            setattr(reference_press, "fit_direct", original)
