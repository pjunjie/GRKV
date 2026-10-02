# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Joint GQA attention residuals and analytic derivatives; no model autograd."""

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import torch


def flatten_heads(x):
    # H,G,Q,D -> Q,H*G*D, the exact o_proj input order.
    return x.permute(2, 0, 1, 3).reshape(x.shape[2], -1)


def unflatten_heads(x, heads, groups):
    return x.reshape(x.shape[0], heads, groups, -1).permute(1, 2, 0, 3)


@dataclass
class Observations:
    queries: torch.Tensor  # H,G,Q,D
    positions: torch.Tensor  # Q, original logical query positions
    target: torch.Tensor  # Q,H*G*D
    suffix_lse: torch.Tensor  # H,G,Q; -inf for virtual history
    suffix_value: torch.Tensor  # H,G,Q,D
    weights: torch.Tensor  # Q, total six regardless of coverage/token length

    def to(self, device):
        return Observations(*(getattr(self, name).to(device) for name in self.__dataclass_fields__))


def suffix_statistics(queries, keys, values, visible):
    """Fixed native suffix sufficient statistics, with explicit causal visibility."""
    logits = torch.einsum("hgqd,hsd->hgqs", queries, keys) / sqrt(keys.shape[-1])
    logits = logits.masked_fill(~visible[:, None], -torch.inf)
    lse = logits.logsumexp(-1)
    weights = (logits - lse[..., None]).exp().nan_to_num(0)
    return lse, torch.einsum("hgqs,hsd->hgqd", weights, values)


class AttentionMap:
    def __init__(self, obs, positions, valid, mutable):
        self.obs = obs
        self.visible = valid[:, None, :] & (positions[:, None, :] <= obs.positions[None, :, None])
        self.mutable = mutable & valid

    def state(self, keys, values):
        q = self.obs.queries
        logits = torch.einsum("hgqd,hsd->hgqs", q, keys) / sqrt(keys.shape[-1])
        logits = logits.masked_fill(~self.visible[:, None], -torch.inf)
        log_z = torch.logaddexp(logits.logsumexp(-1), self.obs.suffix_lse)
        if not bool(torch.isfinite(log_z).all()):
            raise RuntimeError("attention observation has no visible key")
        weights = (logits - log_z[..., None]).exp()
        out = torch.einsum("hgqs,hsd->hgqd", weights, values)
        out += (self.obs.suffix_lse - log_z).exp()[..., None] * self.obs.suffix_value
        return flatten_heads(out), weights, out

    def derivatives(self, keys, values, kind) -> tuple[Callable, Callable]:
        _, weights, out = self.state(keys, values)
        w = weights * self.mutable[:, None, None, :]
        q = self.obs.queries

        def forward(delta):
            if kind == "value":
                y = torch.einsum("hgqs,hsd->hgqd", w, delta)
            else:
                dl = torch.einsum("hgqd,hsd->hgqs", q, delta) / sqrt(keys.shape[-1])
                weighted = w * dl
                y = torch.einsum("hgqs,hsd->hgqd", weighted, values) - weighted.sum(-1)[..., None] * out
            return flatten_heads(y)

        def adjoint(residual):
            u = unflatten_heads(residual, q.shape[0], q.shape[1])
            if kind == "value":
                return torch.einsum("hgqs,hgqd->hsd", w, u)
            uv = torch.einsum("hgqd,hsd->hgqs", u, values)
            uy = (u * out).sum(-1)[..., None]
            return torch.einsum("hgqs,hgqd->hsd", w * (uv - uy), q) / sqrt(keys.shape[-1])

        if kind not in ("key", "value"):
            raise ValueError(kind)
        return forward, adjoint


class ResidualMetric:
    """A rectangular square root L with L^T L = .5 I + .5 W_o^T W_o / g^2."""

    def __init__(self, projection, weights, post):
        self.projection = projection
        self.root_weight = weights.sqrt()[:, None]
        self.post = post
        self.gain = (projection.square().sum() / projection.shape[1]).sqrt()
        if not bool(self.gain > 0):
            raise ValueError("zero projection gain")

    def forward(self, residual):
        e = residual * self.root_weight
        if not self.post:
            return e
        return torch.cat((e, (e @ self.projection.T) / self.gain), -1) / sqrt(2)

    def adjoint(self, residual):
        if not self.post:
            return residual * self.root_weight
        pre, post = residual.split((self.projection.shape[1], self.projection.shape[0]), -1)
        return (pre + (post @ self.projection) / self.gain) * self.root_weight / sqrt(2)


def conjugate_gradient(operator, rhs, max_iterations=16, relative_tolerance=1e-6):
    """Single joint SPD system. Explicit residual is reported even at the cap."""
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    norm = torch.linalg.vector_norm(rhs)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("nonfinite CG right hand side")
    iterations = 0
    if float(norm) > 0:
        rr = (residual * residual).sum()
        for iterations in range(1, max_iterations + 1):
            ad = operator(direction)
            denominator = (direction * ad).sum()
            if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
                raise RuntimeError("nonpositive/nonfinite CG curvature")
            alpha = rr / denominator
            x = x + alpha * direction
            residual = residual - alpha * ad
            next_rr = (residual * residual).sum()
            if float(next_rr.sqrt() / norm) <= relative_tolerance:
                break
            direction = residual + (next_rr / rr) * direction
            rr = next_rr
    true_residual = rhs - operator(x)
    relative = float(torch.linalg.vector_norm(true_residual) / norm) if float(norm) else 0.0
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(true_residual).all()):
        raise RuntimeError("nonfinite CG solution")
    return x, dict(
        iterations=iterations,
        relative_residual=relative,
        converged=relative <= relative_tolerance,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
    )


def ridge_update(forward, adjoint, metric, residual, ridge, max_iterations=16, tolerance=1e-6):
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    def a(delta):
        return metric.forward(forward(delta))

    def at(dual):
        return adjoint(metric.adjoint(dual))

    rhs = metric.forward(residual)
    dual, record = conjugate_gradient(lambda u: a(at(u)) + ridge * u, rhs, max_iterations, tolerance)
    delta = at(dual)
    normal_residual = at(a(delta) - rhs) + ridge * delta
    record["normal_equation_relative_residual"] = float(
        torch.linalg.vector_norm(normal_residual) / torch.linalg.vector_norm(at(rhs)).clamp_min(1e-30)
    )
    return delta, record
