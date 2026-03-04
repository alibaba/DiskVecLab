#!/usr/bin/env python3
import argparse
from typing import Tuple

import numpy as np


def get_bin_metadata(bin_file) -> Tuple[int, int]:
    """读取前 8 字节的 npts, ndims"""
    array = np.fromfile(file=bin_file, dtype=np.uint32, count=2)
    return int(array[0]), int(array[1])


def bin_to_numpy(dtype, bin_file) -> np.ndarray:
    """按照 bin 格式读取数据: [uint32 npts][uint32 ndims][data...]"""
    npts, ndims = get_bin_metadata(bin_file)
    return np.fromfile(file=bin_file, dtype=dtype, offset=8).reshape(npts, ndims)


def numpy_to_bin(array: np.ndarray, out_file: str):
    """把 numpy 数组写成 bin 格式: [uint32 npts][uint32 ndims][data...]"""
    npts = np.uint32(array.shape[0])
    ndims = np.uint32(array.shape[1])

    with open(out_file, "wb") as f:
        f.write(npts.tobytes())
        f.write(ndims.tobytes())
        f.write(array.astype(np.float32, copy=False).tobytes())


def u8bin_to_fbin(in_file: str, out_file: str, normalize: bool = False):
    """
    把 u8bin 转成 fbin

    默认行为：直接把 uint8 值转成 float32（0~255 -> 0.0~255.0）
    如果 normalize=True，则会除以 255.0 归一化到 [0,1]
    """
    print(f"Reading u8bin: {in_file}")
    u8_array = bin_to_numpy(np.uint8, in_file)
    print(f"  shape = {u8_array.shape}, dtype = {u8_array.dtype}")

    if normalize:
        f_array = u8_array.astype(np.float32) / 255.0
        print("  normalize: divide by 255.0 → [0,1]")
    else:
        f_array = u8_array.astype(np.float32)
        print("  no normalize: keep numeric values (0~255)")

    print(f"Writing fbin: {out_file}")
    numpy_to_bin(f_array, out_file)
    print("Done.")


def fbin_to_u8bin(in_file: str, out_file: str, normalize: bool = False):
    """
    把 fbin 转成 u8bin

    默认行为：直接把 float32 值转成 uint8（0.0~255.0 -> 0~255）
    如果 normalize=True，则会乘以 255.0 从 [0,1] 反映射到 [0,255]
    """
    print(f"Reading fbin: {in_file}")
    f_array = bin_to_numpy(np.float32, in_file)
    print(f"  shape = {f_array.shape}, dtype = {f_array.dtype}")

    if normalize:
        u8_array = (f_array * 255.0).astype(np.uint8)
        print("  denormalize: multiply by 255.0 → [0,255]")
    else:
        u8_array = f_array.astype(np.uint8)
        print("  no denormalize: cast float to uint8 (clamped)")

    print(f"Writing u8bin: {out_file}")
    npts = np.uint32(u8_array.shape[0])
    ndims = np.uint32(u8_array.shape[1])

    with open(out_file, "wb") as f:
        f.write(npts.tobytes())
        f.write(ndims.tobytes())
        f.write(u8_array.astype(np.uint8, copy=False).tobytes())
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert between u8bin and fbin files (same layout, different dtype)."
    )
    parser.add_argument("input", help="input file path (.u8bin or .fbin)")
    parser.add_argument("output", help="output file path (.fbin or .u8bin)")
    parser.add_argument(
        "-c", "--conversion",
        choices=["u8_to_f", "f_to_u8"],
        help="conversion type: u8_to_f (u8bin→fbin) or f_to_u8 (fbin→u8bin). Auto-detect by extension if not specified.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="u8_to_f: divide by 255.0 to map [0,255] → [0,1]; f_to_u8: multiply by 255.0 to map [0,1] → [0,255]",
    )

    args = parser.parse_args()
    
    # 自动检测转换类型
    conversion = args.conversion
    if conversion is None:
        if args.input.endswith(".u8bin"):
            conversion = "u8_to_f"
        elif args.input.endswith(".fbin"):
            conversion = "f_to_u8"
        else:
            print("Error: Cannot auto-detect conversion type. Please use -c/--conversion to specify.")
            return
    
    if conversion == "u8_to_f":
        u8bin_to_fbin(args.input, args.output, normalize=args.normalize)
    elif conversion == "f_to_u8":
        fbin_to_u8bin(args.input, args.output, normalize=args.normalize)


if __name__ == "__main__":
    main()
