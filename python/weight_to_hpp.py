#!/usr/bin/env python3
# export_tshm_to_hpp.py
#
# Dumps a PyTorch state_dict into a C++ header with int8 quantized weights.
# This is mainly meant for embedded / ESP-style builds where we want
# everything constexpr-ish and simple.
#
# Usage:
#   python tshm_hpp.py best_tshm_mfcc.pth tshm_weights.hpp

import sys
import torch
import numpy as np


def quantize_tensor(arr: np.ndarray):
    """
    Simple symmetric int8 quantization.
    Nothing fancy here — max-abs scaling only.
    """
    max_abs = float(np.max(np.abs(arr)))

    if max_abs == 0.0:
        # corner case: all zeros, keep scale sane
        scale = 1.0
        q = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = max_abs / 127.0
        q = np.round(arr / scale).astype(np.int8)

    return q, float(scale)


def name_to_c(name: str) -> str:
    """
    Convert a PyTorch parameter name into something
    that is legal (and readable enough) in C.
    """
    return name.replace(".", "_").replace("/", "_")


def write_array(f, cname: str, q: np.ndarray):
    """
    Emit a flat int8 array.
    We keep line width reasonable so diffs are readable.
    """
    f.write(f"static const int8_t {cname}_q[{q.size}] = {{\n")

    flat = q.flatten()
    for i in range(0, flat.size, 16):
        chunk = flat[i:i + 16]
        line = ", ".join(str(int(v)) for v in chunk)
        if i + 16 < flat.size:
            f.write("  " + line + ",\n")
        else:
            f.write("  " + line + "\n")

    f.write("};\n\n")


def write_shape(f, cname: str, shape):
    """
    Emit shape metadata as a simple int array.
    """
    dims = ", ".join(str(int(s)) for s in shape)
    f.write(
        f"static const int {cname}_shape[{len(shape)}] = {{{dims}}};\n\n"
    )


def main():
    if len(sys.argv) < 3:
        print("Usage: export_tshm_to_hpp.py <model.pth> <out.hpp>")
        sys.exit(1)

    src_path = sys.argv[1]
    out_path = sys.argv[2]

    # Load checkpoint on CPU — no reason to touch GPU here
    sd = torch.load(src_path, map_location="cpu")

    if not isinstance(sd, dict):
        raise RuntimeError("Unexpected checkpoint format (expected state_dict).")

    # In most of our runs, this is already model.state_dict().
    # If we ever wrap it differently, this is the place to fix it.
    state = sd

    with open(out_path, "w") as f:
        f.write("#ifndef TSHM_WEIGHTS_HPP\n")
        f.write("#define TSHM_WEIGHTS_HPP\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Auto-generated file — do not edit by hand\n")
        f.write("// Quantized int8 weights + per-tensor scales\n\n")

        for name, tensor in state.items():
            # Skip anything weird (optimizer state, metadata, etc.)
            if not isinstance(tensor, torch.Tensor):
                continue

            np_arr = tensor.cpu().numpy()
            cname = name_to_c(name)

            q, scale = quantize_tensor(np_arr)

            write_array(f, cname, q)

            # scale is stored separately so dequant can stay simple
            f.write(
                f"static const float {cname}_scale = {scale:.18e}f;\n\n"
            )

            write_shape(f, cname, np_arr.shape)

        f.write("#endif // TSHM_WEIGHTS_HPP\n")

    print("Exported weights to:", out_path)


if __name__ == "__main__":
    main()
