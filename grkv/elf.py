# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read CUDA ELF sections without modifying instruction or allocation bytes."""

import hashlib
import struct


def sections(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64 cubin")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, strings = struct.unpack_from("<HHH", data, 58)
    rows = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    start, length = rows[strings][4:6]
    names = data[start : start + length]
    result = {}
    for row in rows:
        name = names[row[0] :].split(b"\0", 1)[0].decode()
        result[name] = dict(type=row[1], flags=row[2], offset=row[4], size=row[5])
    return result


def executable_identity(data):
    result = {}
    for name, section in sections(data).items():
        if "debug" in name:
            if section["flags"] & 6:
                raise ValueError("Debug section unexpectedly allocated/executable")
            continue
        start, size = section["offset"], section["size"]
        value = b"" if section["type"] == 8 else data[start : start + size]
        result[name] = dict(
            type=section["type"], flags=section["flags"], size=size, sha256=hashlib.sha256(value).hexdigest()
        )
    return result
