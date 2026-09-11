#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free GGUF inspector.

Reads only the header + tensor-info section of a .gguf file (no weight data is
loaded) and prints the architecture and a tally of per-tensor quantization types.

Use it to confirm which quant types a model actually contains -- e.g. to check
whether a "Q4_K_M" file includes Q5_K tensors that the OpenVINO GenAI GGUF
reader does not yet natively dequantize.

    python gguf_inspect.py path/to/model.gguf
"""
import struct
import sys

# ggml tensor (quant) types -> name. Mirrors ggml's enum.
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}

# Quant types the OVMS / OpenVINO GenAI gguf reader dequantizes natively.
# (Q5_K added by this change; everything else falls back to gguflib's
#  gguf_tensor_to_f16, which returns null -> "gguf_tensor_to_f16 failed".)
NATIVE = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K"}

# GGUF metadata value types.
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = range(6)
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(6, 13)

_SCALAR = {
    T_UINT8: "B", T_INT8: "b", T_UINT16: "H", T_INT16: "h",
    T_UINT32: "I", T_INT32: "i", T_FLOAT32: "f", T_BOOL: "B",
    T_UINT64: "Q", T_INT64: "q", T_FLOAT64: "d",
}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.f.read(n))

    def u32(self):
        return self.raw("I")[0]

    def u64(self):
        return self.raw("Q")[0]

    def string(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def skip_value(self, vtype):
        if vtype in _SCALAR:
            self.raw(_SCALAR[vtype])
        elif vtype == T_STRING:
            self.string()
        elif vtype == T_ARRAY:
            elem = self.u32()
            count = self.u64()
            for _ in range(count):
                self.skip_value(elem)
        else:
            raise ValueError(f"unknown metadata value type {vtype}")

    def read_value(self, vtype):
        if vtype == T_STRING:
            return self.string()
        if vtype in _SCALAR:
            return self.raw(_SCALAR[vtype])[0]
        self.skip_value(vtype)
        return None


def inspect(path):
    with open(path, "rb") as f:
        r = Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            sys.exit(f"Not a GGUF file (magic={magic!r})")
        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()

        arch = None
        file_type = None
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            val = r.read_value(vtype)
            if key == "general.architecture":
                arch = val
            elif key == "general.file_type":
                file_type = val

        tally = {}
        unsupported = {}
        for _ in range(tensor_count):
            _name = r.string()
            n_dims = r.u32()
            r.raw(f"{n_dims}Q")  # dims
            ttype = r.u32()
            r.u64()  # offset
            tname = GGML_TYPE.get(ttype, f"UNKNOWN({ttype})")
            tally[tname] = tally.get(tname, 0) + 1
            if tname not in NATIVE:
                unsupported[tname] = unsupported.get(tname, 0) + 1

    print(f"File:           {path}")
    print(f"GGUF version:   {version}")
    print(f"Architecture:   {arch}")
    print(f"file_type:      {file_type}")
    print(f"Tensor count:   {tensor_count}")
    print("\nQuant type tally:")
    for name in sorted(tally, key=lambda k: -tally[k]):
        flag = "" if name in NATIVE else "   <-- NOT natively dequantized"
        print(f"  {name:12} {tally[name]:5}{flag}")

    print()
    if unsupported:
        print("RESULT: file contains quant types the GenAI reader cannot handle:")
        for name in sorted(unsupported, key=lambda k: -unsupported[k]):
            print(f"  - {name} ({unsupported[name]} tensors)")
        print("These are what trigger 'gguf_tensor_to_f16 failed'.")
    else:
        print("RESULT: all tensor quant types are natively supported.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python gguf_inspect.py path/to/model.gguf")
    inspect(sys.argv[1])
