#!/usr/bin/env python3
import os
import struct
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Tuple, Union

import numpy as np

# ==================== User config ====================

SRC_PATH = ...

BASE_OUT_PATH = ...
QUERY_OUT_PATH = ...

BASE_N = 1_000      # 1K
QUERY_N = 0      # 2K

NUM_WORKERS = min(8, cpu_count())

# header: [num_vectors(uint32), dim(uint32)]
HEADER_BYTES = 8

# =====================================================

def infer_dtype_from_path(path: str) -> np.dtype:
    """Infer dtype by extension: .fbin -> float32, .u8bin -> uint8."""
    p = path.lower()
    if p.endswith(".fbin"):
        return np.dtype(np.float32)
    if p.endswith(".u8bin") or p.endswith(".bin"):
        return np.dtype(np.uint8)
    raise RuntimeError(
        f"Cannot infer dtype from extension for: {path}\n"
        "Supported: .fbin (float32), .u8bin (uint8)."
    )

def read_header(path: str) -> Tuple[int, int]:
    """Read header, return (num_vectors, dim)."""
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise RuntimeError(f"{path}: header too short")
        # Force little-endian to be explicit
        num, dim = struct.unpack("<II", header)
    return num, dim

def check_file_size(path: str, num: int, dim: int, dtype: Union[np.dtype, type]) -> None:
    """Sanity-check file size matches header + payload."""
    dt = np.dtype(dtype)
    expected = HEADER_BYTES + int(num) * int(dim) * dt.itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        raise RuntimeError(
            f"File size mismatch for {path}\n"
            f"Header says: num={num}, dim={dim}, dtype={dt} => expected_bytes={expected}\n"
            f"Actual file bytes: {actual}\n"
            "Likely wrong dtype/format or corrupted file."
        )

def prepare_out_file(path: str, n: int, dim: int, dtype: Union[np.dtype, type]) -> None:
    """Create output file: write header + reserve payload space."""
    if os.path.exists(path):
        raise RuntimeError(f"Output file already exists: {path}")
    dt = np.dtype(dtype)
    total_bytes = HEADER_BYTES + int(n) * int(dim) * dt.itemsize
    with open(path, "wb") as f:
        f.write(struct.pack("<II", int(n), int(dim)))
        f.truncate(total_bytes)

def _copy_block(args):
    """
    worker: copy src [src_start, src_start+count) to dst [dst_start, dst_start+count)
    All indices are vector indices (not bytes).
    """
    (
        src_path,
        dst_path,
        dim,
        src_start,
        dst_start,
        count,
        dtype,
        header_bytes,
    ) = args

    if count <= 0:
        return

    dt = np.dtype(dtype)
    itemsize = dt.itemsize

    src_offset = header_bytes + int(src_start) * int(dim) * itemsize
    dst_offset = header_bytes + int(dst_start) * int(dim) * itemsize

    src = np.memmap(
        src_path,
        dtype=dt,
        mode="r",
        offset=src_offset,
        shape=(int(count), int(dim)),
        order="C",
    )
    dst = np.memmap(
        dst_path,
        dtype=dt,
        mode="r+",
        offset=dst_offset,
        shape=(int(count), int(dim)),
        order="C",
    )

    dst[:] = src[:]
    dst.flush()

    del src, dst

def parallel_copy(
    src_path: str,
    dst_path: str,
    dim: int,
    total_src_start: int,
    total_count: int,
    dtype: Union[np.dtype, type],
    header_bytes: int = HEADER_BYTES,
    num_workers: int = NUM_WORKERS,
):
    """Copy src range -> dst[0:total_count), chunked by workers."""
    if total_count <= 0:
        return

    num_workers = max(1, int(num_workers))
    chunk_size = (int(total_count) + num_workers - 1) // num_workers

    tasks = []
    for i in range(num_workers):
        local_start = i * chunk_size
        if local_start >= total_count:
            break
        local_end = min(total_count, (i + 1) * chunk_size)
        cnt = local_end - local_start
        src_start = int(total_src_start) + local_start
        dst_start = local_start
        tasks.append(
            (
                src_path,
                dst_path,
                dim,
                src_start,
                dst_start,
                cnt,
                dtype,
                header_bytes,
            )
        )

    print(
        f"[parallel_copy] dtype={np.dtype(dtype)}, src_start={total_src_start}, "
        f"count={total_count}, chunks={len(tasks)}, chunk_size~={chunk_size}"
    )

    if len(tasks) == 1:
        _copy_block(tasks[0])
    else:
        with Pool(processes=len(tasks)) as pool:
            for _ in pool.imap_unordered(_copy_block, tasks):
                pass

def main():
    t0 = datetime.now()
    print(f"[{t0}] Start subset extraction")
    print(f"Source file: {SRC_PATH}")

    src_dtype = infer_dtype_from_path(SRC_PATH)
    num, dim = read_header(SRC_PATH)
    print(f"Header: num_vectors={num}, dim={dim}, dtype={src_dtype}")

    check_file_size(SRC_PATH, num, dim, src_dtype)

    need = BASE_N + QUERY_N
    if num < need:
        raise RuntimeError(f"Dataset too small: {num} < BASE_N+QUERY_N = {need}")

    # Output dtype: infer by output extension; if you want same as src, keep extensions consistent.
    base_dtype = infer_dtype_from_path(BASE_OUT_PATH)
    query_dtype = infer_dtype_from_path(QUERY_OUT_PATH)

    # If you want to *force* same dtype as src regardless of output name, uncomment:
    # base_dtype = src_dtype
    # query_dtype = src_dtype

    if base_dtype != src_dtype or query_dtype != src_dtype:
        print(
            "[WARN] Output dtype differs from source dtype based on file extension.\n"
            f"  src_dtype={src_dtype}, base_dtype={base_dtype}, query_dtype={query_dtype}\n"
            "  If this is unintended, make output extensions match source, or force dtype in code."
        )

    # ========= base subset =========
    print(f"Preparing base subset: N={BASE_N}, out={BASE_OUT_PATH}")
    prepare_out_file(BASE_OUT_PATH, BASE_N, dim, dtype=base_dtype)
    parallel_copy(
        src_path=SRC_PATH,
        dst_path=BASE_OUT_PATH,
        dim=dim,
        total_src_start=0,
        total_count=BASE_N,
        dtype=src_dtype,  # copy uses source dtype
    )
    print("Base subset done.\n")

    # ========= query subset =========
    query_start = BASE_N
    print(
        f"Preparing query subset: N={QUERY_N}, out={QUERY_OUT_PATH}, "
        f"source_range=[{query_start}, {query_start + QUERY_N})"
    )
    prepare_out_file(QUERY_OUT_PATH, QUERY_N, dim, dtype=query_dtype)
    parallel_copy(
        src_path=SRC_PATH,
        dst_path=QUERY_OUT_PATH,
        dim=dim,
        total_src_start=query_start,
        total_count=QUERY_N,
        dtype=src_dtype,  # copy uses source dtype
    )
    print("Query subset done.\n")

    t1 = datetime.now()
    print(f"[{t1}] Finished.")
    print(f"Total time: {t1 - t0}")

if __name__ == "__main__":
    main()
