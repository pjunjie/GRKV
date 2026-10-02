# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


def cpu(tensor):
    return tensor.detach().to("cpu", copy=True)


@dataclass
class Block:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    regression: torch.Tensor

    def to(self, device):
        return Block(*(getattr(self, name).to(device) for name in self.__dataclass_fields__))
