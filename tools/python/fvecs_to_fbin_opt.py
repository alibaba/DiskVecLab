import numpy as np
import os
import argparse
from typing import Optional

def fvecs_to_fbin(
    fvecs_path: str,
    fbin_path: str,
    chunk_mb: int = 256,
    # max_vectors: int | None = None,
    max_vectors: Optional[int] = None,
):
    """
    将 fvecs 文件转换为 DiskANN 风格的 fbin 文件（uint32 n, uint32 d, 后接 float32）。

    针对大文件做了流式处理：
      - 不一次性加载整个 fvecs
      - 按 chunk 读取，chunk 大小由 chunk_mb 控制

    参数:
        fvecs_path: 输入 .fvecs 路径
        fbin_path: 输出 .fbin 路径
        chunk_mb: 每次读取的块大小（MB），默认 256MB
        max_vectors: 只转换前 max_vectors 条向量（可选，None 表示全量）
    """

    if not os.path.exists(fvecs_path):
        raise FileNotFoundError(f"fvecs 文件不存在: {fvecs_path}")

    print(f"📥 输入 fvecs: {fvecs_path}")
    print(f"📤 输出 fbin:  {fbin_path}")
    print(f"⚙️  chunk 大小: {chunk_mb} MB")
    if max_vectors is not None:
        print(f"🔢 最多转换向量数: {max_vectors}")

    total_written = 0

    with open(fvecs_path, "rb") as fin, open(fbin_path, "wb+") as fout:
        # ==== 1. 读取第一个向量，拿到维度 dim ====
        dim_arr = np.fromfile(fin, dtype=np.int32, count=1)
        if dim_arr.size != 1:
            raise ValueError("fvecs 文件内容为空或损坏，无法读取维度。")

        dim = int(dim_arr[0])
        first_vec = np.fromfile(fin, dtype=np.float32, count=dim)
        if first_vec.size != dim:
            raise ValueError("fvecs 文件在第一个向量处截断或损坏。")

        print(f"ℹ️ 检测到向量维度 dim = {dim}")

        # ==== 2. 先写 header，占位 n=0，后续再回填 ====
        header = np.array([0, dim], dtype=np.uint32)
        header.tofile(fout)

        # ==== 3. 写入第一个向量 ====
        first_vec.astype(np.float32, copy=False).tofile(fout)
        total_written = 1

        # 如果只需要 1 个向量就够了，直接回填 header 退出
        if max_vectors is not None and total_written >= max_vectors:
            fout.seek(0)
            np.array([total_written, dim], dtype=np.uint32).tofile(fout)
            print(f"✅ 已写出 {total_written} 个向量（提前结束）。")
            return

        # ==== 4. 按 chunk 流式读取剩余数据 ====
        record_size_bytes = (dim + 1) * 4  # 每条记录: 1 个 int32 + dim 个 float32
        chunk_bytes = chunk_mb * 1024 * 1024
        # 每个 chunk 中大约多少条向量
        vectors_per_chunk = max(1, chunk_bytes // record_size_bytes)

        print(
            f"📦 每次读取约 {vectors_per_chunk} 条向量 "
            f"({chunk_bytes / 1024 / 1024:.1f} MB 原始字节)"
        )

        while True:
            # 从当前文件位置开始，按 int32 读取: 每条向量 (dim+1) 个 int32
            # 注意：第一个向量已经单独处理了，这里读的是“剩余部分”
            need_int32 = vectors_per_chunk * (dim + 1)
            chunk = np.fromfile(fin, dtype=np.int32, count=need_int32)

            if chunk.size == 0:
                break  # EOF

            # 处理尾部不是整条记录的情况（一般不会发生，除非文件损坏）
            if chunk.size % (dim + 1) != 0:
                valid = (chunk.size // (dim + 1)) * (dim + 1)
                if valid == 0:
                    break
                print(
                    f"⚠️ 读取到 {chunk.size} 个 int32，"
                    f"不是 (dim+1) 的整数倍，仅使用前 {valid} 个。"
                )
                chunk = chunk[:valid]

            num_vecs = chunk.size // (dim + 1)

            # 如果设置了 max_vectors，可能不需要这么多
            if max_vectors is not None:
                remain = max_vectors - total_written
                if remain <= 0:
                    break
                if num_vecs > remain:
                    num_vecs = remain
                    chunk = chunk[: num_vecs * (dim + 1)]

            # reshape 成 [num_vecs, dim+1]，丢掉第一列 dim
            chunk = chunk.reshape(num_vecs, dim + 1)
            vecs_int = chunk[:, 1:]  # shape: (num_vecs, dim)

            # 把 int32 的后 dim 列按 bit-level reinterpret 为 float32
            vecs_float = vecs_int.view(np.float32)

            # 直接用 tofile 写 float32 连续块
            vecs_float.tofile(fout)
            total_written += num_vecs

            print(f"   -> 本次写出 {num_vecs} 个向量，累计 {total_written}")

            # 到达上限就停
            if max_vectors is not None and total_written >= max_vectors:
                print("🛑 已达到 max_vectors 限制，停止读取。")
                break

        # ==== 5. 回写 header 中的 n ====
        fout.seek(0)
        np.array([total_written, dim], dtype=np.uint32).tofile(fout)

    print(f"\n✅ 转换完成，总共写出 {total_written} 个向量，维度 {dim}。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .fvecs to .fbin (DiskANN style)")
    parser.add_argument("fvecs_path", help="输入 fvecs 文件路径")
    parser.add_argument("fbin_path", help="输出 fbin 文件路径")
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=256,
        help="每次读取的块大小（MB），默认 256",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=None,
        help="只转换前 N 条向量（可选）",
    )

    args = parser.parse_args()

    fvecs_to_fbin(
        fvecs_path=args.fvecs_path,
        fbin_path=args.fbin_path,
        chunk_mb=args.chunk_mb,
        max_vectors=args.max_vectors,
    )
