#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
bvecs2fvecs: Fast multi-threaded converter from BVECs (uint8) to FVECs (float32).

Formats
-------
- Headered BVECs (standard): each vector is stored as
      int32 dim  +  (uint8[dim])
  repeated for N vectors. This is the default on SIFT1M-style datasets.

- Headerless BVECs: some corpora store raw uint8 vectors without the per-vector
  int32 header. In this case you MUST supply --dim to indicate the vector length.

- FVECs (output): each vector is stored as
      int32 dim  +  (float32[dim])
  repeated for N vectors.

Features
--------
- Auto format detection for headered BVECs (use --mode noheader with --dim for raw files).
- Memory-mapped IO to avoid loading entire files in RAM.
- Multi-threaded conversion with configurable batch size.
- Convert individual files or entire directories (optionally recursive).
- Deterministic output and order-preserving.

Usage
-----
# Convert a single headered .bvecs file
python bvecs2fvecs.py input.bvecs -o output.fvecs

# Convert a headerless file of known dimension
python bvecs2fvecs.py input.bin -o output.fvecs --mode noheader --dim 128

# Convert all .bvecs under a directory to .fvecs in-place
python bvecs2fvecs.py /path/to/dir --recursive

# Convert to another output directory with 8 worker threads
python bvecs2fvecs.py /path/to/dir -O /path/to/out --workers 8 --recursive

Author: ChatGPT
License: MIT
'''
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class BVECFormatError(Exception):
    pass


def _human(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(nbytes)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{nbytes}B"


def detect_bvecs_format(path: Path, *, mode: str = "auto", dim_hint: Optional[int] = None) -> Tuple[bool, int, int]:
    """
    Detect BVECs format.

    Returns:
        (has_header, dim, nvecs)

    Args:
        mode: "auto" | "header" | "noheader"
        dim_hint: if mode is "noheader", this must be provided.
    """
    size = path.stat().st_size
    if size == 0:
        raise BVECFormatError(f"Empty file: {path}")

    if mode not in {"auto", "header", "noheader"}:
        raise ValueError("mode must be one of: auto, header, noheader")

    with open(path, "rb") as f:
        first4 = f.read(4)
    if len(first4) < 4:
        raise BVECFormatError(f"File too small to be BVECs: {path}")

    d_first = int(np.frombuffer(first4, dtype="<i4", count=1)[0])

    def valid_dim(d: int) -> bool:
        return 1 <= d <= 1_000_000  # generous upper bound

    if mode == "header":
        if not valid_dim(d_first):
            raise BVECFormatError(f"Invalid dim header {d_first} in {path}")
        dim = d_first
        rec = 4 + dim  # bytes per vector
        if size % rec != 0:
            raise BVECFormatError(
                f"File size {_human(size)} is not a multiple of record size {rec} in {path}"
            )
        nvecs = size // rec
        return True, dim, nvecs

    if mode == "noheader":
        if dim_hint is None or not isinstance(dim_hint, int) or dim_hint <= 0:
            raise BVECFormatError("--dim must be provided and > 0 for mode=noheader")
        dim = dim_hint
        if size % dim != 0:
            raise BVECFormatError(
                f"File size {_human(size)} is not divisible by dim={dim} for headerless BVECs in {path}"
            )
        nvecs = size // dim
        return False, dim, nvecs

    # mode == "auto"
    # Heuristic: if the first int looks like a reasonable dim and file size fits (4+dim)*N, treat as headered.
    if valid_dim(d_first) and size % (4 + d_first) == 0:
        dim = d_first
        nvecs = size // (4 + dim)
        return True, dim, nvecs

    # Otherwise require dim_hint to treat as headerless.
    if dim_hint is None:
        raise BVECFormatError(
            f"Auto-detect failed. {path} does not look like headered BVECs (first dim={d_first}, size={_human(size)}). "
            f"Provide --mode noheader and --dim to convert headerless files."
        )
    dim = dim_hint
    if size % dim != 0:
        raise BVECFormatError(
            f"Auto-detect fell back to headerless but size {_human(size)} not divisible by dim={dim} in {path}."
        )
    nvecs = size // dim
    return False, dim, nvecs


def default_batch_size(dim: int, target_mb: int = 64) -> int:
    """Choose a batch size (vectors per chunk) to keep per-batch float buffers around target_mb."""
    bytes_per_vec = dim * 4  # output float32 bytes (input uint8 negligible)
    if bytes_per_vec <= 0:
        return 1024
    bs = max(1, int((target_mb * 1024 * 1024) // bytes_per_vec))
    return max(1, bs)


def convert_bvecs_file(
    in_path: Path,
    out_path: Path,
    *,
    mode: str = "auto",
    dim: Optional[int] = None,
    workers: int = max(1, os.cpu_count() or 1),
    batch: Optional[int] = None,
    quiet: bool = False,
) -> Tuple[int, int]:
    """Convert a single BVECs file to FVECs.

    Returns:
        (nvecs, dim)
    """
    has_header, d, n = detect_bvecs_format(in_path, mode=mode, dim_hint=dim)
    if not quiet:
        print(f"[+] {in_path.name}: detected has_header={has_header}, dim={d}, nvecs={n}")

    if n == 0:
        raise BVECFormatError(f"No vectors found in {in_path}")

    if batch is None:
        batch = default_batch_size(d, target_mb=64)

    # Prepare input memmap
    if has_header:
        dtype_in = np.dtype([("d", "<i4"), ("v", "u1", d)])
        arr_in = np.memmap(in_path, dtype=dtype_in, mode="r", shape=(n,))
    else:
        dtype_in = np.dtype(("u1", d))  # shape (n, d)
        arr_in = np.memmap(in_path, dtype=dtype_in, mode="r", shape=(n,))

    # Prepare output memmap
    dtype_out = np.dtype([("d", "<i4"), ("v", "<f4", d)])
    arr_out = np.memmap(out_path, dtype=dtype_out, mode="w+", shape=(n,))

    # Write all dims (broadcast); safe once
    arr_out["d"][:] = d

    # Prepare chunks
    chunks: List[Tuple[int, int]] = []
    for start in range(0, n, batch):
        end = min(n, start + batch)
        chunks.append((start, end))

    # Progress state
    done = 0
    lock = Lock()

    def work(s_e: Tuple[int, int]) -> Tuple[int, int]:
        s, e = s_e
        if has_header:
            u8_block = arr_in["v"][s:e]           # (m, d) uint8
        else:
            u8_block = arr_in[s:e]                # (m, d) uint8
        # Convert to float32
        f_block = u8_block.astype(np.float32, copy=True)  # allocate per-batch float buffer
        arr_out["v"][s:e] = f_block
        return s, e

    if not quiet:
        print(f"[+] Converting with {workers} worker(s), batch={batch} vecs (~{_human(batch * d * 4)} per batch)")

    if workers <= 1 or len(chunks) == 1:
        for s_e in chunks:
            work(s_e)
            if not quiet:
                done += (s_e[1] - s_e[0])
                print(f"\r    {done}/{n} vectors ({done / max(1,n) * 100:.1f}%)", end="", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work, ce) for ce in chunks]
            for fut in as_completed(futures):
                s, e = fut.result()
                if not quiet:
                    with lock:
                        done += (e - s)
                        print(f"\r    {done}/{n} vectors ({done / max(1,n) * 100:.1f}%)", end="", flush=True)

    if not quiet:
        print("\r    Done." + " " * 32)
        print(f"[+] Wrote {out_path} ({_human(out_path.stat().st_size)})")

    # Flush maps
    arr_out.flush()
    del arr_in, arr_out

    return n, d


def gather_input_files(inputs: List[Path], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            if recursive:
                for q in p.rglob("*"):
                    if q.is_file() and q.suffix.lower() in {".bvecs", ".bin", ".bv"}:
                        files.append(q)
            else:
                for q in p.glob("*"):
                    if q.is_file() and q.suffix.lower() in {".bvecs", ".bin", ".bv"}:
                        files.append(q)
        else:
            # glob pattern
            for q in Path().glob(str(p)):
                if q.is_file():
                    files.append(q)
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            uniq.append(f)
            seen.add(rp)
    return uniq


def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        description="Convert BVECs (uint8) to FVECs (float32) with multi-threaded, memory-mapped IO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="Input files/directories/globs")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output file (only valid for single input file). If omitted, writes alongside input with .fvecs")
    ap.add_argument("-O", "--out-dir", type=Path, default=None,
                    help="Output directory for converted files (preserves base filename)." )
    ap.add_argument("--mode", choices=["auto", "header", "noheader"], default="auto",
                    help="BVECs input mode. Use 'noheader' for raw uint8 files (requires --dim)." )
    ap.add_argument("--dim", type=int, default=None,
                    help="Vector dimension for headerless BVECs (required for --mode noheader). Ignored otherwise.")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                    help="Number of worker threads.")
    ap.add_argument("--batch", type=int, default=None,
                    help="Batch size (vectors per chunk). Default auto-scales to ~64MB per batch.")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into subdirectories when inputs include directories.")
    ap.add_argument("-q", "--quiet", action="store_true", help="Reduce console output.")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    inputs = gather_input_files(args.inputs, recursive=args.recursive)
    if not inputs:
        print("No input files found.", file=sys.stderr)
        return 2

    # Validate output argument combinations
    if args.output and len(inputs) != 1:
        print("--output/-o may only be used with a single input file.", file=sys.stderr)
        return 2

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    total_vecs = 0
    last_dim = None

    for in_path in inputs:
        if args.output:
            out_path = args.output
        elif args.out_dir:
            out_path = args.out_dir / (in_path.stem + ".fvecs")
        else:
            out_path = in_path.with_suffix(".fvecs")

        if not args.quiet:
            print(f"[*] Converting {in_path} -> {out_path}")

        nvecs, dim = convert_bvecs_file(
            in_path,
            out_path,
            mode=args.mode,
            dim=args.dim,
            workers=args.workers,
            batch=args.batch,
            quiet=args.quiet,
        )
        total_vecs += nvecs
        last_dim = dim

    if not args.quiet:
        print(f"[✓] Converted {len(inputs)} file(s), total {total_vecs} vectors, dim={last_dim}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
