# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from experiments.critical_adakv_postproj_grkv.operators import Observations


def promote_observations(observations, device, dtype):
    values = []
    for name in observations.__dataclass_fields__:
        value = getattr(observations, name)
        values.append(value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype))
    return Observations(*values)
