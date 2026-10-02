# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from experiments.critical_adakv_grkv.press import CriticalAdaGRKVPress, tensor_sha


@dataclass
class NativeBoundaryPress(CriticalAdaGRKVPress):
    """Keep original native selection for T>32; disclose untouched short inputs."""

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if keys.shape[2] > self.window_size:
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)
        tokens = keys.shape[2]
        module.masked_key_indices = tuple(torch.empty(0, dtype=torch.long, device=keys.device) for _ in range(3))
        self.layer_records.append(
            dict(
                layer=int(module.layer_idx),
                context_tokens=tokens,
                returned_path="native_short_untouched",
                compression_applied=False,
                native_short_path_supported=False,
                kept=keys.shape[1] * tokens,
                kept_per_head=[tokens] * keys.shape[1],
                keep_sha256=tensor_sha(torch.ones(keys.shape[:3], dtype=torch.bool, device=keys.device)),
                regression_slots=0,
                fixed=keys.shape[1] * tokens,
                changed=0,
                fixed_unchanged=True,
                unchosen_unchanged=True,
            )
        )
        return keys, values
