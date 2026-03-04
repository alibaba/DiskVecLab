
from __future__ import annotations
import numpy as np
import struct
from typing import Tuple


def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]

def read_fvecs(filename):
    return read_ivecs(filename).view("float32")

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

import os
import numpy as np
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor

def _record_dtype(vec_dtype: np.dtype, d: int) -> np.dtype:
    """
    每条记录: int32 dim + vec[d]
    align=False 确保不插 padding（很关键，保证 itemsize 精确等于 4 + d*itemsize）
    """
    return np.dtype([("dim", "<i4"), ("vec", vec_dtype, (d,))], align=False)

def numpy_to_vecs_fast(
    filename: Union[str, os.PathLike],
    m: np.ndarray,
    dtype: Optional[Union[np.dtype, type, str]] = None,
    *,
    chunk_rows: int = 200_000,
    buffer_mb: int = 64,
    pipeline: bool = True,
) -> None:
    """
    高速写 fvecs/ivecs/bvecs（dim(int32)+vector）格式。
    - 支持 m 是 np.memmap（不会整块加载）
    - 分块写，避免一次性占用巨大内存

    参数建议：
    - dim=128,float32: 一行 4 + 128*4 = 516B
      chunk_rows=200k => ~103MB 输出块，比较均衡
    """
    m = np.asarray(m)  # 支持 memmap / ndarray
    if m.ndim != 2:
        raise ValueError(f"m must be 2D, got shape={m.shape}")

    n, d = m.shape
    vec_dtype = np.dtype(dtype if dtype is not None else m.dtype)

    if vec_dtype not in (np.dtype(np.float32), np.dtype(np.uint32), np.dtype(np.uint8)):
        raise ValueError(f"unsupported dtype: {vec_dtype} (expected float32/uint32/uint8)")

    rec_dt = _record_dtype(vec_dtype, d)
    expected_itemsize = 4 + d * vec_dtype.itemsize
    if rec_dt.itemsize != expected_itemsize:
        raise RuntimeError(
            f"record dtype has padding? itemsize={rec_dt.itemsize}, expected={expected_itemsize}"
        )

    def pack_chunk(chunk: np.ndarray) -> bytes:
        # 保证连续，避免写入时出现隐式拷贝/慢路径
        chunk_c = np.ascontiguousarray(chunk, dtype=vec_dtype)
        out = np.empty(chunk_c.shape[0], dtype=rec_dt)
        out["dim"] = d
        out["vec"] = chunk_c
        return out.tobytes(order="C")

    print(f"Writing File - {filename}")
    bufsize = max(1, int(buffer_mb)) * 1024 * 1024

    with open(filename, "wb", buffering=bufsize) as f:
        if not pipeline:
            # 单线程：已经比逐行 pack 快很多
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                f.write(pack_chunk(m[start:end]))
        else:
            # 流水线：线程预先打包下一块，同时主线程写上一块
            with ThreadPoolExecutor(max_workers=1) as ex:
                start0 = 0
                end0 = min(chunk_rows, n)
                fut = ex.submit(pack_chunk, m[start0:end0])

                start = end0
                while start < n:
                    end = min(start + chunk_rows, n)
                    data = fut.result()
                    f.write(data)
                    fut = ex.submit(pack_chunk, m[start:end])
                    start = end

                # 写最后一块
                f.write(fut.result())

    print(f"\t{filename} wrote (n={n}, d={d}, dtype={vec_dtype}, chunk_rows={chunk_rows})")

def numpy_to_vecs(filename, m, dtype):
    numpy_to_vecs_fast(filename, m, dtype)
    # print(f"Writing File - {filename}")
    # n, d = m.shape
    # if dtype == np.float32:
    #     fmt = f"{d}f"
    # elif dtype == np.uint32:
    #     fmt = f"{d}I"
    # elif dtype == np.uint8:
    #     fmt = f"{d}B"
    # else:
    #     raise ValueError(f"invalid dtype: {type(dtype)}")
    # with open(filename, "wb") as f:
    #     for i in range(n):
    #         f.write(struct.pack("i", d))
    #         bin = struct.pack(fmt, *m[i])
    #         f.write(bin)
    # print(f"\t{filename} wrote")

from typing import Tuple, Union, Optional
import os
import struct
import numpy as np


def get_bin_metadata(bin_file: Union[str, os.PathLike]) -> Tuple[int, int]:
    """
    Read header of .bin/.fbin/.u8bin-like file:
      [npts(uint32), ndims(uint32)] then raw data.
    Returns: (npts, ndims)
    """
    with open(bin_file, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"File too small (need >= 8 bytes header): {bin_file}")
        # Most vector bin formats are little-endian uint32
        npts, ndims = struct.unpack("<II", header)
    return int(npts), int(ndims)


def bin_to_numpy(
    dtype: Union[np.dtype, str],
    bin_file: Union[str, os.PathLike],
    *,
    mode: str = "r",
    validate_file_size: bool = True,
    order: str = "C",
) -> np.memmap:
    """
    Memory-mapped load. Returns np.memmap with shape (npts, ndims).

    Parameters
    ----------
    dtype : numpy dtype or str
        Data dtype for vectors (e.g. np.float32, np.uint8).
    mode : str
        'r' (read-only) recommended; 'r+' if you intend to modify.
    validate_file_size : bool
        If True, sanity-check file size matches header.
    order : str
        'C' is typical.

    Notes
    -----
    - This does NOT read all data into RAM; it pages in on demand.
    - Slicing like X[i] or X[i:j] is efficient; operations that touch all data
      will still read a lot from disk.
    """
    dtype = np.dtype(dtype)
    npts, ndims = get_bin_metadata(bin_file)

    if npts <= 0 or ndims <= 0:
        raise ValueError(f"Bad header: npts={npts}, ndims={ndims}, file={bin_file}")

    offset = 8  # 2 * uint32

    if validate_file_size:
        file_size = os.path.getsize(bin_file)
        expected = offset + (np.int64(npts) * np.int64(ndims) * np.int64(dtype.itemsize))
        if file_size < expected:
            raise ValueError(
                f"File truncated? size={file_size}, expected_at_least={int(expected)} "
                f"(npts={npts}, ndims={ndims}, dtype={dtype})"
            )
        # 有些格式可能在末尾追加额外内容；你若希望严格相等可改成 != 检查
        # if file_size != expected: ...

    # Key: use memmap + shape to avoid reading into memory
    mm = np.memmap(
        bin_file,
        dtype=dtype,
        mode=mode,
        offset=offset,
        shape=(npts, ndims),
        order=order,
    )
    return mm


def numpy_to_bin(array, out_file):
    shape = np.shape(array)
    npts = np.uint32(shape[0])
    ndims = np.uint32(shape[1])
    f = open(out_file, "wb")
    f.write(npts.tobytes())
    f.write(ndims.tobytes())
    f.write(array.tobytes())
    f.close()

def numpy_to_pure(array, out_file):
    f = open(out_file, "wb")
    f.write(array.tobytes())
    f.close()

def read_gt_bin_file(gt_file) -> Tuple[np.ndarray[int], np.ndarray[float]]:
    """
    Return ids and distances to queries
    """
    nq, K = get_bin_metadata(gt_file)
    ids = np.fromfile(file=gt_file, dtype=np.uint32, offset=8, count=nq * K).reshape(
        nq, K
    )
    dists = np.fromfile(
        file=gt_file, dtype=np.float32, offset=8 + nq * K * 4, count=nq * K
    ).reshape(nq, K)
    return ids, dists


def calculate_recall(
        result_set_indices: np.ndarray[int],
        truth_set_indices: np.ndarray[int],
        recall_at: int = 5,
) -> float:
    found = 0
    for i in range(0, result_set_indices.shape[0]):
        result_set_set = set(result_set_indices[i][0:recall_at])
        truth_set_set = set(truth_set_indices[i][0:recall_at])
        found += len(result_set_set.intersection(truth_set_set))
    return found / (result_set_indices.shape[0] * recall_at)

def get_sampled_data(m, num_sampled) -> np.ndarray:
    n, d = m.shape
    sample_size = min(num_sampled, n)
    sample_indices = np.random.choice(range(n), size=sample_size, replace=False)
    sampled_data = m[sample_indices, :]
    return sampled_data
