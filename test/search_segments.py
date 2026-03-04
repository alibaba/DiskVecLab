"""
This module wraps your three-stage workflow — Partition → Generate Config → Build,
plus Search — into a clean Python API without a CLI.

Usage sketch
------------
from starling_runner import (
    GlobalConfig, DatasetSpec,
    NaturalParams, KMeansParams, SPANNParams, ELPISParams,
    BuildParams, SearchParams,
    create_and_build_experiment, search_experiment,
)

# 1) Global + dataset
G = GlobalConfig(
    starling_root="/root/paper/DiskAnnPQ/starling",
    scripts_dir="/root/paper/DiskAnnPQ/starling/scripts",
    dataset_root="/root/paper",
)

DS = DatasetSpec(
    name="sift1m", dim=128, dtype="float",
    base_file="/root/paper/sift1m/sift_base.fvecs",
    query_file="/root/paper/sift1m/sift_query.fvecs",
    work_dir="/root/paper/sift1m",
)

# 2) Partition params (choose exactly one)
PP = KMeansParams(
    out_dir="/root/paper/sift1m/data_split/0911knnd2",
    sample_ratio=0.05, num_shards=40, dup_per_vec=2,
    use_partition_data_exec=False,  # use run_data_partition knn
)

# 3) Build params
BP = BuildParams(
    dtype="float",
    split_name="0911knnd2",
    generate_config_dataset="sift1m",
    config_local_path="/root/paper/DiskAnnPQ/starling/scripts/config_local.sh",
)

# 4) Create & build (partition + generate_config + fixed build sequence)
create_and_build_experiment(
    name="0911knnd2",
    global_cfg=G,
    dataset=DS,
    partition=PP,
    build=BP,
    force=False,
)

# 5) Search on an already built experiment
SP = SearchParams(
    build_type="release", mode="search_split", algo="knn",
    aio_max=20971520,
    dataset_env_script="config_dataset_auto_spann.sh",
    dataset_env_func="dataset_sift1m_spann_shard_0",
    config_local_overrides={"MEM_USE_FREQ": "1", "GP_USE_FREQ": "1", "FREQ_MEM_L": "0"},
)
search_experiment(
    name="0911knnd2",
    global_cfg=G,
    dataset=DS,
    search=SP,
)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, Tuple, Dict, Any, List
import os
import re
import json
import time
import shutil
import pathlib
import subprocess
import numpy as np
# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class GlobalConfig:
    starling_root: str
    scripts_dir: str
    dataset_root: str
    # external binaries/tools (override if your paths differ)
    bin_run_data_partition: str | None = None
    bin_partition_data: str | None = None
    bin_sptag_ssdserving: str | None = None
    bin_elpis: str | None = None
    tool_vecs_to_pure: str | None = None
    tool_generate_ids: str | None = None

    def __post_init__(self) -> None:
        # Sensible defaults, assuming usual layout under DiskAnnPQ/
        root = os.path.abspath(os.path.join(self.starling_root, os.pardir))
        self.bin_run_data_partition = self.bin_run_data_partition or os.path.join(
            self.starling_root, "release/tests/utils/run_data_partition")
        self.bin_partition_data = self.bin_partition_data or os.path.join(
            self.starling_root, "release/tests/utils/partition_data")
        self.bin_sptag_ssdserving = self.bin_sptag_ssdserving or os.path.join(
            root, "SPTAG/Release/ssdserving")
        self.bin_elpis = self.bin_elpis or os.path.join(
            root, "ELPIS/code/build/ELPIS")
        self.tool_vecs_to_pure = self.tool_vecs_to_pure or os.path.join(
            root, "tools/python/vecs_to_pure.py")
        self.tool_generate_ids = self.tool_generate_ids or os.path.join(
            root, "tools/python/generate_ids.py")

@dataclass
class DatasetSpec:
    name: str
    dim: int
    dtype: Literal["float", "uint8"]
    base_file: str   # e.g., /root/paper/sift1m/sift_base.fvecs|fbin|bvecs
    query_file: str
    work_dir: str    # e.g., /root/paper/sift1m
    # Optional alternative-format paths for stages needing specific formats
    base_file_fvecs: Optional[str] = None
    base_file_bvecs: Optional[str] = None
    base_file_fbin: Optional[str] = None
    base_file_bin: Optional[str] = None
    query_file_fvecs: Optional[str] = None
    query_file_bvecs: Optional[str] = None
    query_file_fbin: Optional[str] = None
    query_file_bin: Optional[str] = None

# --- Partition method params ---
@dataclass
class NaturalParams:
    split_name: str            
    out_dir: str
    num_shards: int
    input_fmt: Optional[Literal["fvecs", "bvecs"]] = None

@dataclass
class KMeansParams:
    split_name: str
    out_dir: str
    sample_ratio: float
    num_shards: int
    dup_per_vec: int
    use_partition_data_exec: bool = True  # False→run_data_partition knn; True→partition_data

@dataclass
class SPANNParams:
    split_name: str
    out_dir: str
    template: Literal["default-1m", "default-1b"]
    overrides: Dict[str, Any] | None = None
    config_path: Optional[str] = None  # where to write the ini (defaults to experiment dir)
    index_dir: Optional[str] = None    # optional; defaults to <DS.work_dir>/index/spann/<split_name>

@dataclass
class ELPISParams:
    split_name: str
    out_dir: str
    index_path: str
    dataset_size: int
    timeseries_size: int
    leaf_size: int
    kb: int
    Lb: int
    buffer_size: int
    run_query: bool = False
    queries_size: Optional[int] = None
    k: Optional[int] = None
    L: Optional[int] = None
    nprobes: Optional[int] = None

# --- Build & Search ---
@dataclass
class BuildParams:
    generate_config_dataset: str    # e.g., "sift1m"
    config_local_path: str          # starling/scripts/config_local.sh
    dtype: Optional[Literal["float", "uint8"]] = None 

    # fixed sequence (first round)
    first_round_variants: Tuple[Tuple[str, str], ...] = (
        ("release", "build"),
        ("release", "build_mem"),
        # ("release", "gp"), # disable freqs
        # ("release", "freq"),
    )
    # fixed sequence (second round)
    second_round_variants: Tuple[Tuple[str, str], ...] = (
        # ("release", "build_mem"),  # disable freqs
        ("release", "gp"),
    )
    first_round_env: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {"MEM_USE_FREQ": "0", "GP_USE_FREQ": "0", "FREQ_MEM_L": "0"}
    )
    second_round_env: Dict[str, str] = dataclasses.field(
        # default_factory=lambda: {"MEM_USE_FREQ": "1", "GP_USE_FREQ": "1"}
        default_factory=lambda: {"MEM_USE_FREQ": "0", "GP_USE_FREQ": "0", "FREQ_MEM_L": "0"}
    )

@dataclass
class SearchParams:
    build_type: str = "release"
    mode: str = "search_split"
    algo: str = "knn"
    aio_max: int = 20971520
    shard_id: int = 0
    config_local_overrides: Dict[str, str] = dataclasses.field(default_factory=dict)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

class RunError(RuntimeError):
    pass

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

def _mkdir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _write_text(path: str, s: str) -> None:
    _mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def _append_text(path: str, s: str) -> None:
    _mkdir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(s)

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _normalize_for_compare(obj: Any) -> Any:
    """递归地把 tuple → list，dict 里的值也一起处理，用于做参数等价性比较。"""
    if isinstance(obj, dict):
        return {k: _normalize_for_compare(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_compare(v) for v in obj]
    return obj


def _run_cmd(cmd: str | List[str], cwd: Optional[str], log_path: str,
             env: Optional[Dict[str, str]] = None, shell: bool = False) -> None:
    """Run a command, tee stdout/stderr to a log file, raise on failure.
    - If `shell=False` and cmd is list, use exec form.
    - If `shell=True`, cmd must be str and is executed by bash -lc.
    """
    _mkdir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as log:
        banner = f"\n===== RUN {_now()} =====\nCWD: {cwd or os.getcwd()}\nCMD: {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n\n"
        log.write(banner)
        log.flush()
        if shell:
            proc = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, stdout=log, stderr=log)
        else:
            if isinstance(cmd, str):
                cmd_list = cmd.split()
            else:
                cmd_list = cmd
            proc = subprocess.Popen(cmd_list, cwd=cwd, stdout=log, stderr=log, env=env)
        ret = proc.wait()
        log.write(f"\n===== EXIT CODE {ret} =====\n")
        log.flush()
        if ret != 0:
            raise RunError(f"Command failed (code={ret}). See log: {log_path}")

def _save_run_json(exp_dir: str, meta: Dict[str, Any]) -> None:
    path = os.path.join(exp_dir, "run.json")
    _write_text(path, json.dumps(meta, indent=2))


def _symlink_force(src: str, dst: str) -> None:
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except FileNotFoundError:
        pass
    _mkdir(os.path.dirname(dst))
    os.symlink(src, dst)


def _infer_vecfmt(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext in {".fvecs", ".fbin"}:  # treat .fbin as float vectors
        return "fvecs" if ext == ".fvecs" else "fbin"
    if ext in {".bvecs", ".bin"}:   # treat .bin as uint8 vectors
        return "bvecs" if ext == ".bvecs" else "bin"
    return ext.lstrip(".")

def _spann_pick_path_and_type(paths: list[str | None]) -> tuple[str, str]:
    exts_bin = {".bin", ".fbin", ".u8bin", ".ffbin"}
    exts_xvec = {".fvecs", ".bvecs"}

    # 1) prefer bin-like → DEFAULT
    for p in paths:
        if not p:
            continue
        ext = pathlib.Path(p).suffix.lower()
        if ext in exts_bin:
            return p, "DEFAULT"

    # 2) fallback to xvec-style → XVEC
    for p in paths:
        if not p:
            continue
        ext = pathlib.Path(p).suffix.lower()
        if ext in exts_xvec:
            return p, "XVEC"

    raise ValueError(
        f"None of the candidate paths look like SPANN vector files: {paths!r}"
    )


def _ensure_starling_query_symlink(G: GlobalConfig, DS: DatasetSpec) -> Dict[str, Any]:
    """Starling generate_config sets QUERY_FILE to /root/paper/${DATASET}/${DATASET}_query.fbin.
    Ensure that path exists by symlinking that expected path to DS.query_file if needed.
    We do not change file formats; user must ensure DS.query_file matches expected format.
    """
    expected_dir = os.path.join(G.dataset_root, DS.name)
    expected_path = os.path.join(expected_dir, f"{DS.name}_query.fbin")
    _mkdir(expected_dir)

    info: Dict[str, Any] = {"expected_query_path": expected_path, "source_query": DS.query_file}

    if os.path.lexists(expected_path):
        if os.path.islink(expected_path):
            cur = os.readlink(expected_path)
            if cur != DS.query_file:
                os.remove(expected_path)
                os.symlink(DS.query_file, expected_path)
                info["action"] = "updated_symlink"
            else:
                info["action"] = "kept_symlink"
        else:
            info["action"] = "exists_non_symlink"
        return info

    try:
        os.symlink(DS.query_file, expected_path)
        info["action"] = "created_symlink"
    except OSError as e:
        info["action"] = "error"
        info["error"] = str(e)
    return info

# -----------------------------------------------------------------------------
# SPANN config templates & rendering
# -----------------------------------------------------------------------------

_SPANN_TEMPLATE_1M = """
[Base]
ValueType=Float
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=128
VectorPath=/root/paper/sift1m/sift_base.fvecs
VectorType=XVEC
QueryPath=/root/paper/sift1m/sift_query.fvecs
QueryType=XVEC
WarmupPath=/root/paper/sift1m/sift_query.fvecs
WarmupType=XVEC
TruthPath=/root/paper/sift1m/sift_groundtruth.ivecs
TruthType=XVEC
IndexDirectory=/root/paper/sift1m/spann

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=30000
SamplesNumber=1000
SaveBKT=false
SelectThreshold=0
SplitFactor=4
SplitThreshold=100000
Ratio=0.00004
NumberOfThreads=128
BKTLambdaFactor=-1

[BuildHead]
isExecute=true
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=8192
MaxCheckForRefineGraph=8192
RefineIterations=3
NumberOfThreads=128
BKTLambdaFactor=-1

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=8
PostingPageLimit=12500
NumberOfThreads=128
MaxCheck=8192
TmpDir=/tmp/
enableOutputDataSplit=true
splitDirectory=/root/paper/sift1m/data_split/spann
outputEmptyReplicaID=true
""".strip()

_SPANN_TEMPLATE_1B = """
[Base]
ValueType=uint8
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=128
VectorPath=/root/paper/sift1b/bigann_base.bvecs
VectorType=XVEC
QueryPath=/root/paper/sift1b/bigann_query.bvecs
QueryType=XVEC
WarmupPath=/root/paper/sift1b/bigann_query.bvecs
WarmupType=XVEC
TruthPath=/root/paper/sift1b/sift_query_learn_gt100
TruthType=XVEC
IndexDirectory=/root/paper/sift1b/spann

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=30000000
SamplesNumber=1000
SaveBKT=false
SelectThreshold=20
SplitFactor=4
SplitThreshold=100000000
Ratio=0.00000004
NumberOfThreads=128
BKTLambdaFactor=-1

[BuildHead]
isExecute=true
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=8192
MaxCheckForRefineGraph=8192
RefineIterations=3
NumberOfThreads=128
BKTLambdaFactor=-1

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=8
PostingPageLimit=12500000
NumberOfThreads=128
MaxCheck=8192
TmpDir=/tmp/
enableOutputDataSplit=true
splitDirectory=/root/paper/sift1b/data_split/spann
outputEmptyReplicaID=true
""".strip()


def _render_spann_config(template_name: str, overrides: Dict[str, Any] | None) -> str:
    if template_name == "default-1m":
        txt = _SPANN_TEMPLATE_1M
    elif template_name == "default-1b":
        txt = _SPANN_TEMPLATE_1B
    else:
        raise ValueError(f"Unknown SPANN template: {template_name}")

    lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("#")]
    txt = "\n".join(lines)

    if not overrides:
        return txt

    global_kv: Dict[str, Any] = {}
    sect_kv: Dict[str, Dict[str, Any]] = {}

    for k, v in overrides.items():
        if "." in k:
            sect, key = k.split(".", 1)
            sect_kv.setdefault(sect, {})[key] = v
        else:
            global_kv[k] = v

    def replace_in_section(txt: str, section: str, key: str, sval: str) -> str:
        sec_pat = re.compile(rf"(?ms)(\[{re.escape(section)}\].*?)(?=\n\[|\Z)")
        m = sec_pat.search(txt)
        if not m:
            return txt 
        block = m.group(1)
        key_pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*.*$", re.M)
        if key_pat.search(block):
            block2 = key_pat.sub(f"{key}={sval}", block)
        else:
            block2 = block + f"\n{key}={sval}"
        return txt[:m.start(1)] + block2 + txt[m.end(1):]

    # 1) 先处理 section-aware 覆盖
    for sect, kv in sect_kv.items():
        for key, v in kv.items():
            txt = replace_in_section(txt, sect, key, str(v))

    # 2) 再处理全局覆盖（替换所有同名 key）
    for k, v in global_kv.items():
        sval = str(v)
        pat = re.compile(rf"^\s*{re.escape(k)}\s*=\s*.*$", re.M)
        if pat.search(txt):
            txt = pat.sub(f"{k}={sval}", txt)
        else:
            if k in {"splitDirectory", "ReplicaCount", "PostingPageLimit"}:
                txt = re.sub(r"(?ms)(\[BuildSSDIndex\][^\[]*)$", r"\1\n" + f"{k}={sval}", txt)
            elif k in {"VectorPath", "QueryPath", "WarmupPath", "TruthPath", "Dim", "ValueType"}:
                txt = re.sub(r"(?ms)(\[Base\][^\[]*)$", r"\1\n" + f"{k}={sval}", txt)
            else:
                txt = re.sub(r"(?ms)(\[SelectHead\][^\[]*)$", r"\1\n" + f"{k}={sval}", txt)

    return txt


def _partition_natural(name: str, G: GlobalConfig, DS: DatasetSpec, P: NaturalParams,
                       exp_dir: str) -> Dict[str, Any]:
    """
    Natural partition on your setup can consume **fvecs** (float) or **bvecs** (uint8).

    - 优先使用 P.input_fmt 显式指定的格式；
    - 如果未指定，则根据 DatasetSpec.dtype 自动推断：
      - dtype == "float"  → fvecs
      - dtype == "uint8" → bvecs
    """
    # Ensure output directory exists
    _mkdir(P.out_dir)

    # 1) 决定输入格式
    fmt = P.input_fmt
    if fmt is None:
        # 根据 dtype 猜一个合理的默认值
        fmt = "fvecs" if DS.dtype == "float" else "bvecs"

    if fmt not in ("fvecs", "bvecs"):
        raise ValueError(
            "Natural partition only supports input_fmt in {'fvecs', 'bvecs'} "
            f"(got {fmt!r})."
        )

    # 2) 根据 fmt 选 base 向量文件
    candidates: List[str] = []

    if fmt == "fvecs":
        if DS.base_file_fvecs:
            candidates.append(DS.base_file_fvecs)
        if DS.base_file.lower().endswith(".fvecs"):
            candidates.append(DS.base_file)
        missing_hint_attr = "base_file_fvecs"
        missing_ext = ".fvecs"
    else:  # fmt == "bvecs"
        if DS.base_file_bvecs:
            candidates.append(DS.base_file_bvecs)
        if DS.base_file.lower().endswith(".bvecs"):
            candidates.append(DS.base_file)
        missing_hint_attr = "base_file_bvecs"
        missing_ext = ".bvecs"

    base_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not base_path:
        raise FileNotFoundError(
            f"Natural partition needs a {fmt} base file. "
            f"Please set DatasetSpec.{missing_hint_attr} to an existing {missing_ext} path, "
            f"or make DatasetSpec.base_file end with {missing_ext}."
        )

    # 3) 调用 run_data_partition
    cmd = [
        G.bin_run_data_partition,
        "natural",
        base_path,
        P.out_dir,
        str(P.num_shards),
        fmt,
    ]
    log = os.path.join(exp_dir, "partition", f"natural_{_now()}.log")
    _mkdir(os.path.dirname(log))
    _run_cmd(cmd, cwd=None, log_path=log)

    return {
        "method": "natural",
        "out_dir": P.out_dir,
        "log": log,
        "input_fmt": fmt,
        "base_used": base_path,
    }



def _partition_kmeans(name: str, G: GlobalConfig, DS: DatasetSpec, P: KMeansParams,
                      exp_dir: str) -> Dict[str, Any]:
    log = os.path.join(exp_dir, "partition", f"kmeans_{_now()}.log")
    # Ensure output directory exists
    _mkdir(P.out_dir)
    if P.use_partition_data_exec:
        # /.../partition_data <dtype> <base> <outdir> <sample> <shards> <dup>
        cmd = [
            G.bin_partition_data, DS.dtype, DS.base_file, P.out_dir,
            str(P.sample_ratio), str(P.num_shards), str(P.dup_per_vec)
        ]
        _run_cmd(cmd, cwd=None, log_path=log)
    else:
        # /.../run_data_partition knn <base> <out_prefix> <dtype> <sample> <shards> <dup>
        # out_dir used as out_prefix
        cmd = [
            G.bin_run_data_partition, "knn", DS.base_file, P.out_dir, DS.dtype,
            str(P.sample_ratio), str(P.num_shards), str(P.dup_per_vec)
        ]
        _run_cmd(cmd, cwd=None, log_path=log)
    return {"method": "kmeans", "out_dir": P.out_dir, "log": log,
            "sample_ratio": P.sample_ratio, "num_shards": P.num_shards, "dup": P.dup_per_vec}


def _partition_spann(name: str, G: GlobalConfig, DS: DatasetSpec, P: SPANNParams,
                     exp_dir: str) -> Dict[str, Any]:
    # Ensure output directory exists for split files
    _mkdir(P.out_dir)

    # Compute index directory (avoid global default); create it
    index_dir = P.index_dir or os.path.join(DS.work_dir, "index", "spann", P.split_name)
    _mkdir(index_dir)

    # ---------------- 关键改动开始：自动选择 bin/fbin or fvecs/bvecs ----------------
    # 优先用 bin/fbin（DEFAULT），没有再用 fvecs/bvecs（XVEC）
    base_path, vec_type = _spann_pick_path_and_type([
        DS.base_file,
        DS.base_file_fbin,
        DS.base_file_bin,
        DS.base_file_fvecs,
        DS.base_file_bvecs,
    ])
    query_path, query_type = _spann_pick_path_and_type([
        DS.query_file,
        DS.query_file_fbin,
        DS.query_file_bin,
        DS.query_file_fvecs,
        DS.query_file_bvecs,
    ])
    warmup_path = query_path
    warmup_type = query_type
    # ---------------- 关键改动结束 ----------------

    # Render config.ini and run SPTAG ssdserving with it
    overrides = dict(P.overrides or {})
    overrides.setdefault("splitDirectory", P.out_dir)
    overrides.setdefault("IndexDirectory", index_dir)
    overrides.setdefault("Dim", DS.dim)
    # ValueType from DS.dtype
    overrides.setdefault("ValueType", "Float" if DS.dtype == "float" else "uint8")

    # 只有用户没有在 overrides 里手动指定时，我们才自动塞默认值
    overrides.setdefault("VectorPath", base_path)
    overrides.setdefault("QueryPath", query_path)
    overrides.setdefault("WarmupPath", warmup_path)
    overrides.setdefault("VectorType", vec_type)      # XVEC or DEFAULT
    overrides.setdefault("QueryType", query_type)     # XVEC or DEFAULT
    overrides.setdefault("WarmupType", warmup_type)   # XVEC or DEFAULT

    # Choose template
    cfg_txt = _render_spann_config(P.template, overrides)
    cfg_path = P.config_path or os.path.join(exp_dir, "partition", "spann_config.ini")
    _write_text(cfg_path, cfg_txt)

    log = os.path.join(exp_dir, "partition", f"spann_{_now()}.log")
    _run_cmd([G.bin_sptag_ssdserving, cfg_path], cwd=None, log_path=log)
    return {"method": "spann", "out_dir": P.out_dir, "config": cfg_path, "log": log}



def _convert_elpis_subshards_to_dtype(out_dir: str, target_dtype: str) -> Dict[str, Any]:
    """
    Convert ELPIS partition subshards from float32 to target dtype (e.g., uint8).
    Skips conversion if target_dtype is float or if already converted.
    """
    if target_dtype.lower() in ["float", "float32"]:
        return {"converted": False, "reason": "target dtype is float, no conversion needed"}
    
    import glob
    subshard_files = sorted(glob.glob(os.path.join(out_dir, "_subshard-*.bin")))
    subshard_files = [f for f in subshard_files if not f.endswith("_ids_uint32.bin")]
    
    if not subshard_files:
        return {"converted": False, "reason": "no subshard files found"}
    
    print(f"[ELPIS subshard conversion] Converting {len(subshard_files)} files to {target_dtype}")
    
    # Map dtype string to numpy dtype
    dtype_map = {
        "uint8": np.uint8,
        "uint32": np.uint32,
        "int32": np.int32,
        "float32": np.float32,
        "float": np.float32,
    }
    target_np_dtype = dtype_map.get(target_dtype.lower())
    if target_np_dtype is None:
        return {"converted": False, "reason": f"unknown target dtype: {target_dtype}"}
    
    converted_files = []
    for fpath in subshard_files:
        try:
            # Preserve original float shards before overwriting
            backup_path = fpath + ".float.backup"
            src_path = backup_path if os.path.exists(backup_path) else fpath

            # If backup does not exist, create it first
            if not os.path.exists(backup_path):
                os.rename(fpath, backup_path)
                src_path = backup_path

            # Read float32 subshard with header from backup
            with open(src_path, "rb") as f:
                header = f.read(8)
                npts, ndims = np.frombuffer(header, dtype=np.uint32)
                float_data = np.fromfile(f, dtype=np.float32, count=npts * ndims).reshape(npts, ndims)
            
            # Convert to target dtype
            if target_np_dtype == np.uint8:
                # Directly convert float to uint8, clip to valid range
                converted_data = np.clip(float_data, 0, 255).astype(np.uint8)
            elif target_np_dtype == np.uint32:
                converted_data = np.clip(float_data, 0, 2**32 - 1).astype(np.uint32)
            else:
                converted_data = float_data.astype(target_np_dtype)
            
            # Write converted data to the original path name (fpath)
            with open(fpath, "wb") as f:
                f.write(np.uint32(npts).tobytes())
                f.write(np.uint32(ndims).tobytes())
                f.write(converted_data.tobytes())
            
            converted_files.append(fpath)
            print(f"  Converted: {os.path.basename(fpath)}")
        except Exception as e:
            print(f"  ERROR converting {fpath}: {e}")
            return {"converted": False, "error": str(e), "files_converted": len(converted_files)}
    
    return {"converted": True, "files_converted": len(converted_files), "target_dtype": target_dtype, "backups_kept": True}


def _elpis_ids_already_generated(out_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """Check whether each _subshard-*.bin already has a matching _ids file."""
    import glob

    subshard_files = sorted(glob.glob(os.path.join(out_dir, "_subshard-*.bin")))
    subshard_files = [f for f in subshard_files if not f.endswith("_ids_uint32.bin")]

    if not subshard_files:
        return False, {"reason": "no subshard files found"}

    missing: List[str] = []
    for fpath in subshard_files:
        leaf = os.path.basename(fpath)[len("_subshard-"):-len(".bin")]
        ids_path = os.path.join(out_dir, f"_subshard-{leaf}_ids_uint32.bin")
        if not os.path.exists(ids_path):
            missing.append(ids_path)

    if missing:
        return False, {
            "expected": len(subshard_files),
            "missing_count": len(missing),
            "missing_examples": missing[:3],
        }

    return True, {"expected": len(subshard_files), "ids_found": len(subshard_files)}


def _partition_elpis(name: str, G: GlobalConfig, DS: DatasetSpec, P: ELPISParams,
                     exp_dir: str) -> Dict[str, Any]:
    part_dir = os.path.join(exp_dir, "partition")
    _mkdir(part_dir)
    base_pure = os.path.join(part_dir, f"{DS.name}_base.pure")
    query_pure = os.path.join(part_dir, f"{DS.name}_query.pure")

    # Convert base/query to .pure (needs input format string) - skip if already exists
    if not os.path.exists(base_pure) or not os.path.exists(query_pure):
        base_fmt = _infer_vecfmt(DS.base_file_fvecs)
        query_fmt = _infer_vecfmt(DS.query_file_fvecs)
        log_conv = os.path.join(part_dir, f"elpis_convert_{_now()}.log")
        _run_cmd([
            "python", G.tool_vecs_to_pure, DS.base_file_fvecs, base_fmt, base_pure
        ], cwd=None, log_path=log_conv)
        _run_cmd([
            "python", G.tool_vecs_to_pure, DS.query_file_fvecs, query_fmt, query_pure
        ], cwd=None, log_path=log_conv)
    else:
        log_conv = None  # Not run

    # Build ELPIS index / split - skip if index already exists
    print(f"[ELPIS] Building index at {P.index_path} (if not exists)...")
    if not os.path.exists(P.index_path):
        log_build = os.path.join(part_dir, f"elpis_build_{_now()}.log")
        # Ensure output directory exists for split files
        _mkdir(P.out_dir)

        cmd_build = [
            G.bin_elpis,
            "--dataset", base_pure,
            "--dataset-size", str(P.dataset_size),
            "--index-path", P.index_path,
            "--timeseries-size", str(P.timeseries_size),
            "--leaf-size", str(P.leaf_size),
            "--kb", str(P.kb),
            "--Lb", str(P.Lb),
            "--mode", "0",
            "--buffer-size", str(P.buffer_size),
            "--split-path", P.out_dir,
        ]
        _run_cmd(cmd_build, cwd=None, log_path=log_build)
    else:
        log_build = None  # Not run

    # generate_ids: must use float splits + original float base to avoid precision loss
    # do this BEFORE any dtype conversion of subshards
    ids_ready, ids_status = _elpis_ids_already_generated(P.out_dir)
    log_ids = None
    base_fbin_guess = DS.base_file if DS.base_file.endswith(".fbin") else (
        os.path.splitext(DS.base_file)[0] + ".fbin")

    if ids_ready:
        ids_info = {"generated_ids": True, "skipped_generate_ids": True, "ids_status": ids_status}
    elif os.path.exists(base_fbin_guess):
        log_ids = os.path.join(part_dir, f"elpis_ids_{_now()}.log")
        _run_cmd([
            "python", G.tool_generate_ids, P.out_dir, base_fbin_guess, "float"
        ], cwd=None, log_path=log_ids)
        ids_info = {"generated_ids": True, "base_fbin": base_fbin_guess, "log": log_ids, "ids_status": ids_status}
    else:
        ids_info = {"generated_ids": False, "reason": f"base fbin not found: {base_fbin_guess}", "ids_status": ids_status}

    # Convert subshards from float to original dtype if needed (after IDs are generated)
    convert_info = {}
    original_dtype = DS.dtype or "float"
    if original_dtype.lower() not in ["float", "float32"]:
        print(f"[ELPIS] Original dataset dtype is {original_dtype}, converting subshards...")
        convert_info = _convert_elpis_subshards_to_dtype(P.out_dir, original_dtype)
        print(f"[ELPIS] Conversion result: {convert_info}")

    if not os.path.exists(P.index_path):
        os.makedirs(os.path.dirname(P.index_path), exist_ok=True)
        
    # optional: query
    qinfo: Dict[str, Any] = {}
    if P.run_query:
        log_q = os.path.join(part_dir, f"elpis_query_{_now()}.log")
        cmd_q = [
            G.bin_elpis,
            "--queries", query_pure,
            "--queries-size", str(P.queries_size or 0),
            "--index-path", P.index_path,
            "--k", str(P.k or 10),
            "--L", str(P.L or 10),
            "--nprobes", str(P.nprobes or 1),
            "--mode", "1",
            "--split-path", P.out_dir,
        ]
        _run_cmd(cmd_q, cwd=None, log_path=log_q)
        qinfo = {"ran_query": True, "log": log_q}

    logs_dict = {"convert": log_conv, "build": log_build}
    if P.run_query:
        logs_dict["query"] = log_q
    return {"method": "elpis", "out_dir": P.out_dir, "logs": logs_dict, "subshard_conversion": convert_info, **ids_info, **qinfo}

# -----------------------------------------------------------------------------
# Build & Search implementations

def _ensure_config_local_symlink(G: GlobalConfig, target_dir: str) -> Dict[str, Any]:
    """Ensure target_dir/config_local.sh exists.
    - If missing, create symlink to <scripts_dir>/config_local.sh
    - If a symlink exists but points elsewhere, refresh it.
    - If a real file exists, leave it intact.
    """
    src = os.path.join(G.scripts_dir, "config_local.sh")
    dst = os.path.join(target_dir, "config_local.sh")
    info: Dict[str, Any] = {"src": src, "dst": dst}
    try:
        if os.path.lexists(dst):
            if os.path.islink(dst):
                cur = os.readlink(dst)
                if cur != src:
                    os.remove(dst)
                    os.symlink(src, dst)
                    info["action"] = "updated_symlink"
                else:
                    info["action"] = "kept_symlink"
            else:
                info["action"] = "exists_non_symlink"
        else:
            os.symlink(src, dst)
            info["action"] = "created_symlink"
    except OSError as e:
        info["action"] = "error"
        info["error"] = str(e)
    return info

def _split_type_from_partition(P: Union[NaturalParams, KMeansParams, SPANNParams, ELPISParams]) -> str:
    if isinstance(P, NaturalParams):
        return "natural"
    if isinstance(P, KMeansParams):
        return "knn"  # Starling scripts use 'knn' for kmeans partition
    if isinstance(P, SPANNParams):
        return "spann"
    if isinstance(P, ELPISParams):
        return "elpis"
    return "unknown"


def _select_dataset_env_script(G: GlobalConfig, split_type: str) -> str:
    cand = [f"config_dataset_auto_{split_type}.sh",
            "config_dataset_auto_spann.sh",  # common fallback
            "config_dataset_auto.sh"]
    for name in cand:
        if os.path.exists(os.path.join(G.scripts_dir, name)):
            return name
    # fall back to spann name (even if missing; search() will attempt to source it and fail clearly)
    return "config_dataset_auto_spann.sh"

def _discover_dataset_func(env_script_path: str, dataset_name: str, split_name: str, split_type: str,
                           preferred_shard_id: Optional[int]) -> Dict[str, Any]:
    """
    从环境脚本中解析可用的 dataset_* 函数名。

    优先级：
      1) dataset_{dataset}_{split_name}_shard_{preferred} 存在 → 选它
      2) 按文件顺序第一个 dataset_{dataset}_{split_name}_shard_*
      3) dataset_{dataset}_{split_type}_shard_{preferred} 存在 → 选它
      4) 按文件顺序第一个 dataset_{dataset}_{split_type}_shard_*
      5) 回退：按文件顺序第一个 dataset_{dataset}_*_shard_*

    返回: {"func_name": str, "candidates": [...], "strategy": str}
    """
    txt = _read_text(env_script_path)

    def find(pattern: str) -> List[Tuple[str, int, int]]:
        # 返回 (函数名, shard_id, 在文件中的位置)
        out: List[Tuple[str, int, int]] = []
        for m in re.finditer(pattern, txt, flags=re.M):
            fname = m.group(1)
            sid = int(m.group(2))
            out.append((fname, sid, m.start()))
        return out

    # 同时支持 `function name {` 和 `name() {` 两种写法
    def pat(mid: str) -> str:
        return rf"^\s*(?:function\s+)?(dataset_{re.escape(dataset_name)}_{re.escape(mid)}_shard_(\d+))\s*(?:\(\))?\s*\{{"

    any_pat = rf"^\s*(?:function\s+)?(dataset_{re.escape(dataset_name)}_[^\s(){{}}]+_shard_(\d+))\s*(?:\(\))?\s*\{{"

    cand_split_name = find(pat(split_name))
    cand_split_type = find(pat(split_type))
    cand_any = find(any_pat)

    def pick(cands: List[Tuple[str, int, int]], label: str) -> Optional[Tuple[str, str]]:
        if not cands:
            return None
        if preferred_shard_id is not None:
            for fn, sid, _ in cands:
                if sid == preferred_shard_id:
                    return fn, f"{label}:preferred_shard_id={preferred_shard_id}"
        # 否则按文件顺序第一个
        fn = sorted(cands, key=lambda t: t[2])[0][0]
        return fn, f"{label}:first_in_file"

    for cands, lbl in ((cand_split_name, "split_name"), (cand_split_type, "split_type")):
        picked = pick(cands, lbl)
        if picked:
            fn, strat = picked
            return {"func_name": fn, "candidates": [c[0] for c in cands], "strategy": strat}

    if cand_any:
        fn = sorted(cand_any, key=lambda t: t[2])[0][0]
        return {"func_name": fn, "candidates": [c[0] for c in cand_any], "strategy": "any:first_in_file"}

    # 实在找不到就给个猜测名，让错误更直观
    guessed = f"dataset_{dataset_name}_{split_name}_shard_{preferred_shard_id or 0}"
    return {"func_name": guessed, "candidates": [], "strategy": "guessed"}


# -----------------------------------------------------------------------------

def _append_env_block(config_local_path: str, marker: str, kv: Dict[str, str]) -> None:
    block = [f"# BEGIN starlingx:{marker}\n"]
    for k, v in kv.items():
        block.append(f"{k}={v}\n")
    block.append("# END starlingx\n\n")
    _append_text(config_local_path, "".join(block))


def _generate_config(G: GlobalConfig, BP: BuildParams, split_name: str, dtype: str, exp_dir: str) -> str:
    log = os.path.join(exp_dir, "build", f"generate_config_{_now()}.log")
    _mkdir(os.path.dirname(log))
    cmd = [
        os.path.join(G.scripts_dir, "auto_generate_config_and_run.sh"),
        "generate_config", BP.generate_config_dataset, split_name, dtype
    ]
    _run_cmd(cmd, cwd=G.scripts_dir, log_path=log)
    return log


def _run_single_variant(
    G: GlobalConfig,
    dataset_name: str,
    split_name: str,
    build_type: str,
    variant: str,
    exp_dir: str,
) -> str:
    log = os.path.join(exp_dir, "build", f"{build_type}_{variant}_{_now()}.log")
    cmd = [
        os.path.join(G.scripts_dir, "auto_generate_config_and_run.sh"),
        "run_benchmarks",
        dataset_name,
        split_name,
        f"{build_type} {variant}",
    ]
    _run_cmd(cmd, cwd=G.scripts_dir, log_path=log)
    return log


def _run_variants(G: GlobalConfig, dataset_name: str, split_name: str,
                  variants: Tuple[Tuple[str, str], ...], exp_dir: str) -> List[str]:
    logs = []
    for build_type, variant in variants:
        log = os.path.join(exp_dir, "build", f"{build_type}_{variant}_{_now()}.log")
        cmd = [
            os.path.join(G.scripts_dir, "auto_generate_config_and_run.sh"),
            "run_benchmarks", dataset_name, split_name, f"{build_type} {variant}"
        ]
        _run_cmd(cmd, cwd=G.scripts_dir, log_path=log)
        logs.append(log)
    return logs


def build_starling(name: str, G: GlobalConfig, DS: DatasetSpec, P: Union[NaturalParams, KMeansParams, SPANNParams, ELPISParams], BP: BuildParams,
                   exp_dir: str) -> Dict[str, Any]:
    _mkdir(os.path.join(exp_dir, "build"))
    # Ensure QUERY_FILE expected path exists (symlink to DS.query_file)
    qlink_info = _ensure_starling_query_symlink(G, DS)
    dtype = BP.dtype or DS.dtype
    split_name = getattr(P, "split_name")
    gen_log = _generate_config(G, BP, split_name, dtype, exp_dir)

    # First round env block
    _append_env_block(BP.config_local_path, f"{name}:first", BP.first_round_env)
    logs1 = _run_variants(G, BP.generate_config_dataset, split_name, BP.first_round_variants, exp_dir)

    # Second round env block
    _append_env_block(BP.config_local_path, f"{name}:second", BP.second_round_env)
    logs2 = _run_variants(G, BP.generate_config_dataset, split_name, BP.second_round_variants, exp_dir)
    # Snapshot env script for search autodiscovery
    split_type = _split_type_from_partition(P)
    env_script_name = _select_dataset_env_script(G, split_type)
    snapshot_dir = os.path.join(exp_dir, "build")
    env_src = os.path.join(G.scripts_dir, env_script_name)
    env_dst = os.path.join(snapshot_dir, "dataset_env.sh")
    if os.path.exists(env_src):
        try:
            shutil.copy2(env_src, env_dst)
        except Exception as e:
            env_dst = f"COPY_FAILED: {e}"
    # Snapshot config_local.sh
    config_local_snap = os.path.join(snapshot_dir, "config_local.snapshot.sh")
    try:
        shutil.copy2(BP.config_local_path, config_local_snap)
    except Exception as e:
        config_local_snap = f"COPY_FAILED: {e}"

    return {"generate_config_log": gen_log, "variant_logs": logs1 + logs2,
            "config_local": BP.config_local_path, "query_symlink": qlink_info,
            "split_type": split_type, "dataset_env_script": env_script_name,
            "snapshots": {"env_script": env_dst, "config_local": config_local_snap}}


def search_starling(name: str, G: GlobalConfig, DS: DatasetSpec, SP: SearchParams,
                     exp_dir: str) -> Dict[str, Any]:
    sdir = os.path.join(exp_dir, "search")
    _mkdir(sdir)

    # Set aio-max-nr
    log_sys = os.path.join(sdir, f"sysctl_{_now()}.log")
    _run_cmd(f"echo {SP.aio_max} | tee /proc/sys/fs/aio-max-nr", cwd=None, log_path=log_sys, shell=True)

    # Append overrides into config_local.sh if provided
    if SP.config_local_overrides:
        _append_env_block(
            os.path.join(G.scripts_dir, "config_local.sh"),
            f"{name}:search", SP.config_local_overrides
        )

    # Source dataset env + run benchmark (scripts rely on relative paths → cwd=scripts_dir)
    log_run = os.path.join(sdir, f"search_{SP.build_type}_{SP.mode}_{SP.algo}_{_now()}.log")
    cmd = (
        f"source {SP.dataset_env_script} && {SP.dataset_env_func} && "
        f"{os.path.join(G.scripts_dir, 'run_benchmark.sh')} {SP.build_type} {SP.mode} {SP.algo}"
    )
    _run_cmd(cmd, cwd=G.scripts_dir, log_path=log_run, shell=True)

    return {"sys_log": log_sys, "search_log": log_run}

def search_starling_auto(name: str, G: GlobalConfig, DS: DatasetSpec, SP: SearchParams,
                          exp_dir: str) -> Dict[str, Any]:
    sdir = os.path.join(exp_dir, "search")
    _mkdir(sdir)

    # sysctl
    log_sys = os.path.join(sdir, f"sysctl_{_now()}.log")
    _run_cmd(f"echo {SP.aio_max} | tee /proc/sys/fs/aio-max-nr", cwd=None, log_path=log_sys, shell=True)

    # read meta to infer split_type/split_name
    run_json_path = os.path.join(exp_dir, "run.json")
    with open(run_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    part = meta.get("steps", {}).get("partition", {})
    split_type = part.get("method")
    if split_type == "kmeans":
        split_type = "knn"
    split_name = part.get("split_name") or split_type  # 有些环境脚本可能没按 split_name 生成函数

    # choose env script
    build_info = meta.get("steps", {}).get("build", {})
    env_snapshot = build_info.get("snapshots", {}).get("env_script")
    if env_snapshot and isinstance(env_snapshot, str) and os.path.exists(env_snapshot):
        env_script_to_source = os.path.abspath(env_snapshot)
        # 为了让 ../release 和 .. 正确解析到 Starling 根/构建目录：cwd 固定在 scripts_dir
        env_cwd = G.scripts_dir
        clink_info = {"action": "cwd_set_to_scripts_dir_for_cmake"}
    else:
        env_script_name = _select_dataset_env_script(G, split_type)
        env_script_to_source = os.path.join(G.scripts_dir, env_script_name)
        env_cwd = G.scripts_dir
        clink_info = {"action": "cwd_set_to_scripts_dir_for_cmake"}

    # --- ELPIS search env (triggered by partition type, not SP.algo) ---
    pparams = meta.get("partition_params", {})
    idx_dir = pparams.get("index_path") or os.path.join(DS.work_dir, "index", "elpis")
    if not str(idx_dir).endswith("/"):
        idx_dir = str(idx_dir) + "/"

    _append_env_block(
        os.path.join(G.scripts_dir, "config_local.sh"),
        f"{name}:search:elpis",
        {
            "ELPIS_INDEX_PATH": f"\"{idx_dir}\"",
            "ELPIS_EF": "10",
            "ELPIS_NWORKER": "0",
            "ELPIS_NPROBES": "1",
            "ELPIS_FLATT": "0",
            "ELPIS_PARALLEL": "1",
        },
    )

    # overrides for config_local
    if SP.config_local_overrides:
        _append_env_block(
            os.path.join(G.scripts_dir, "config_local.sh"),
            f"{name}:search", SP.config_local_overrides
        )



    # 从 env 脚本中发现可用函数（自动处理 shard 从 1/其它数字开始的情况）
    discovery = _discover_dataset_func(env_script_to_source, DS.name, split_name, split_type, SP.shard_id)
    func_name = discovery["func_name"]

    log_run = os.path.join(sdir, f"search_{SP.build_type}_{SP.mode}_{SP.algo}_{_now()}.log")
    cmd = (
        f"source {env_script_to_source} && {func_name} && "
        f"{os.path.join(G.scripts_dir, 'run_benchmark.sh')} {SP.build_type} {SP.mode} {SP.algo}"
    )
    _run_cmd(cmd, cwd=env_cwd, log_path=log_run, shell=True)

    return {
        "sys_log": log_sys,
        "search_log": log_run,
        "env_script": env_script_to_source,
        "func_name": func_name,
        "config_local_link": clink_info,
        "discovery": discovery,
    }


# -----------------------------------------------------------------------------
# Public API (partition + build + search + orchestration)
# -----------------------------------------------------------------------------

def partition(name: str, G: GlobalConfig, DS: DatasetSpec,
              P: Union[NaturalParams, KMeansParams, SPANNParams, ELPISParams],
              exp_dir: str) -> Dict[str, Any]:
    pdir = os.path.join(exp_dir, "partition")
    _mkdir(pdir)

    if isinstance(P, NaturalParams):
        info = _partition_natural(name, G, DS, P, exp_dir)
    elif isinstance(P, KMeansParams):
        info = _partition_kmeans(name, G, DS, P, exp_dir)
    elif isinstance(P, SPANNParams):
        info = _partition_spann(name, G, DS, P, exp_dir)
    elif isinstance(P, ELPISParams):
        info = _partition_elpis(name, G, DS, P, exp_dir)
    else:
        raise ValueError("Unknown partition params type")

    # Maintain a stable path under <dataset>/data_split/<split_name>
    split_name = getattr(P, "split_name")
    canonical = os.path.join(DS.work_dir, "data_split", split_name)
    out_dir = info.get("out_dir", "")
    if out_dir:
        if os.path.abspath(out_dir) == os.path.abspath(canonical):
            info["canonical_split_dir"] = canonical
        else:
            if not os.path.lexists(canonical):
                try:
                    _symlink_force(out_dir, canonical)
                    info["canonical_split_dir"] = canonical
                except FileExistsError as e:
                    info["canonical_symlink_error"] = str(e)
            elif os.path.islink(canonical):
                cur = os.readlink(canonical)
                if cur != out_dir:
                    os.remove(canonical)
                    os.symlink(out_dir, canonical)
                info["canonical_split_dir"] = canonical
            else:
                info["canonical_split_dir_conflict"] = canonical

    info["split_name"] = split_name
    return info


def create_and_build_experiment(
    name: str,
    global_cfg: GlobalConfig,
    dataset: DatasetSpec,
    partition: Union[NaturalParams, KMeansParams, SPANNParams, ELPISParams],
    build: BuildParams,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Partition + generate_config + Starling build sequence, recorded under
    <dataset.work_dir>/experiments/<name>.
    """

    exp_dir = os.path.join(dataset.work_dir, "experiments", name)
    run_json_path = os.path.join(exp_dir, "run.json")

    # force=True：直接删实验目录，重新来
    if force and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    # 初始化 / 读取 meta
    if not os.path.exists(exp_dir):
        _mkdir(exp_dir)
        meta: Dict[str, Any] = {
            "name": name,
            "created_at": _now(),
            "global": asdict(global_cfg),
            "dataset": asdict(dataset),
            "partition_params": asdict(partition),
            "build_params": asdict(build),
            "steps": {},
        }
        _save_run_json(exp_dir, meta)
    else:
        if os.path.exists(run_json_path):
            with open(run_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {
                "name": name,
                "created_at": _now(),
                "global": asdict(global_cfg),
                "dataset": asdict(dataset),
                "partition_params": asdict(partition),
                "build_params": asdict(build),
                "steps": {},
            }
            _save_run_json(exp_dir, meta)

    # 如果参数改了而你没 force，直接给出提示，避免“参数变了但沿用旧状态”
    old_p = meta.get("partition_params")
    old_b = meta.get("build_params")
    new_p = asdict(partition)
    new_b = asdict(build)

    if old_p is not None and _normalize_for_compare(old_p) != _normalize_for_compare(new_p):
        print("Old partition_params:", old_p, flush=True)
        print("New partition_params:", new_p, flush=True)
        raise RuntimeError(
            "Existing experiment has different partition_params; "
            "use a new experiment name or force=True to reset."
        )

    if old_b is not None and _normalize_for_compare(old_b) != _normalize_for_compare(new_b):
        print("Old build_params:", old_b, flush=True)
        print("New build_params:", new_b, flush=True)
        raise RuntimeError(
            "Existing experiment has different build_params; "
            "use a new experiment name or force=True to reset."
        )
    
    meta["global"] = asdict(global_cfg)
    meta["dataset"] = asdict(dataset)
    meta["partition_params"] = new_p
    meta["build_params"] = new_b
    meta.setdefault("created_at", _now())
    meta["updated_at"] = _now()
    steps: Dict[str, Any] = meta.setdefault("steps", {})

    # ------------------------------------------------------------------
    # 1) Partition 阶段：整体作为一个 step，状态为 done / failed
    # ------------------------------------------------------------------
    pstep = steps.get("partition", {})
    if pstep.get("status") != "done":
        pstep["status"] = "running"
        pstep.setdefault("started_at", _now())
        steps["partition"] = pstep
        _save_run_json(exp_dir, meta)

        try:
            part_info = partition_fn(name, global_cfg, dataset, partition, exp_dir)
        except Exception as e:
            pstep["status"] = "failed"
            pstep["error"] = str(e)
            pstep["finished_at"] = _now()
            steps["partition"] = pstep
            _save_run_json(exp_dir, meta)
            raise

        pstep.update(part_info)
        pstep["status"] = "done"
        pstep["finished_at"] = _now()
        steps["partition"] = pstep
        _save_run_json(exp_dir, meta)
    else:
        part_info = pstep  # 已经完成，直接复用

    # ------------------------------------------------------------------
    # 2) Build 阶段：拆成 generate_config + 两轮 variants
    # ------------------------------------------------------------------
    bstep = steps.get("build", {})

    split_name = getattr(partition, "split_name")
    split_type = _split_type_from_partition(partition)

    bstep.setdefault("split_name", split_name)
    bstep.setdefault("split_type", split_type)
    bstep.setdefault("status", "pending")
    bstep.setdefault("first_round_variants", build.first_round_variants)
    bstep.setdefault("second_round_variants", build.second_round_variants)

    steps["build"] = bstep
    _save_run_json(exp_dir, meta)

    if bstep.get("status") != "done":
        if bstep.get("status") != "running":
            bstep["status"] = "running"
            bstep.setdefault("started_at", _now())

        _mkdir(os.path.join(exp_dir, "build"))

        # 确保 Starling 期望的 QUERY_FILE 存在（symlink）
        qlink_info = _ensure_starling_query_symlink(global_cfg, dataset)
        bstep["query_symlink"] = qlink_info
        steps["build"] = bstep
        _save_run_json(exp_dir, meta)

        # ---------------- generate_config 子步骤 ----------------
        gstep = bstep.get("generate_config", {})
        if gstep.get("status") != "done":
            gstep["status"] = "running"
            gstep.setdefault("started_at", _now())
            bstep["generate_config"] = gstep
            steps["build"] = bstep
            _save_run_json(exp_dir, meta)

            try:
                dtype = build.dtype or dataset.dtype
                gen_log = _generate_config(global_cfg, build, split_name, dtype, exp_dir)
            except Exception as e:
                gstep["status"] = "failed"
                gstep["error"] = str(e)
                gstep["finished_at"] = _now()
                bstep["generate_config"] = gstep
                bstep["status"] = "failed"
                bstep["finished_at"] = _now()
                steps["build"] = bstep
                _save_run_json(exp_dir, meta)
                raise

            gstep["log"] = gen_log
            gstep["status"] = "done"
            gstep["finished_at"] = _now()
            bstep["generate_config"] = gstep
            steps["build"] = bstep
            _save_run_json(exp_dir, meta)

        # ---------------- first_round 各个 variant ----------------
        fr = bstep.get("first_round", {})
        fr_variants: Dict[str, Any] = fr.get("variants", {})

        # 只在第一次跑 first_round 时 append env block，避免重复堆 config_local.sh
        if not fr.get("env_block_appended"):
            _append_env_block(
                build.config_local_path,
                f"{name}:first",
                build.first_round_env,
            )
            fr["env_block_appended"] = True

        for build_type, variant in build.first_round_variants:
            vkey = f"{build_type}:{variant}"
            vst = fr_variants.get(vkey, {})
            if vst.get("status") == "done":
                continue  # 已完成，跳过

            fr.setdefault("status", "running")
            fr.setdefault("started_at", _now())
            vst["status"] = "running"
            vst.setdefault("started_at", _now())
            fr_variants[vkey] = vst
            fr["variants"] = fr_variants
            bstep["first_round"] = fr
            steps["build"] = bstep
            _save_run_json(exp_dir, meta)

            try:
                log = _run_single_variant(
                    global_cfg,
                    build.generate_config_dataset,
                    split_name,
                    build_type,
                    variant,
                    exp_dir,
                )
            except Exception as e:
                vst["status"] = "failed"
                vst["error"] = str(e)
                vst["finished_at"] = _now()
                fr_variants[vkey] = vst
                fr["variants"] = fr_variants
                fr["status"] = "failed"
                fr["finished_at"] = _now()
                bstep["first_round"] = fr
                bstep["status"] = "failed"
                bstep["finished_at"] = _now()
                steps["build"] = bstep
                _save_run_json(exp_dir, meta)
                raise

            vst["log"] = log
            vst["status"] = "done"
            vst["finished_at"] = _now()
            fr_variants[vkey] = vst
            fr["variants"] = fr_variants
            if all(v.get("status") == "done" for v in fr_variants.values()):
                fr["status"] = "done"
                fr["finished_at"] = _now()
            bstep["first_round"] = fr
            steps["build"] = bstep
            _save_run_json(exp_dir, meta)

        # ---------------- second_round 各个 variant ----------------
        sr = bstep.get("second_round", {})
        sr_variants: Dict[str, Any] = sr.get("variants", {})

        if not sr.get("env_block_appended"):
            _append_env_block(
                build.config_local_path,
                f"{name}:second",
                build.second_round_env,
            )
            sr["env_block_appended"] = True

        for build_type, variant in build.second_round_variants:
            vkey = f"{build_type}:{variant}"
            vst = sr_variants.get(vkey, {})
            if vst.get("status") == "done":
                continue

            sr.setdefault("status", "running")
            sr.setdefault("started_at", _now())
            vst["status"] = "running"
            vst.setdefault("started_at", _now())
            sr_variants[vkey] = vst
            sr["variants"] = sr_variants
            bstep["second_round"] = sr
            steps["build"] = bstep
            _save_run_json(exp_dir, meta)

            try:
                log = _run_single_variant(
                    global_cfg,
                    build.generate_config_dataset,
                    split_name,
                    build_type,
                    variant,
                    exp_dir,
                )
            except Exception as e:
                vst["status"] = "failed"
                vst["error"] = str(e)
                vst["finished_at"] = _now()
                sr_variants[vkey] = vst
                sr["variants"] = sr_variants
                sr["status"] = "failed"
                sr["finished_at"] = _now()
                bstep["second_round"] = sr
                bstep["status"] = "failed"
                bstep["finished_at"] = _now()
                steps["build"] = bstep
                _save_run_json(exp_dir, meta)
                raise

            vst["log"] = log
            vst["status"] = "done"
            vst["finished_at"] = _now()
            sr_variants[vkey] = vst
            sr["variants"] = sr_variants
            if all(v.get("status") == "done" for v in sr_variants.values()):
                sr["status"] = "done"
                sr["finished_at"] = _now()
            bstep["second_round"] = sr
            steps["build"] = bstep
            _save_run_json(exp_dir, meta)

        # ---------------- 构建完成后，snapshot env 脚本和 config_local ----------------
        snapshot_dir = os.path.join(exp_dir, "build")
        env_script_name = _select_dataset_env_script(global_cfg, split_type)
        env_src = os.path.join(global_cfg.scripts_dir, env_script_name)
        env_dst = os.path.join(snapshot_dir, "dataset_env.sh")
        if os.path.exists(env_src):
            try:
                shutil.copy2(env_src, env_dst)
            except Exception as e:
                env_dst = f"COPY_FAILED: {e}"
        config_local_snap = os.path.join(snapshot_dir, "config_local.snapshot.sh")
        try:
            shutil.copy2(build.config_local_path, config_local_snap)
        except Exception as e:
            config_local_snap = f"COPY_FAILED: {e}"

        bstep["dataset_env_script"] = env_script_name
        bstep["snapshots"] = {
            "env_script": env_dst,
            "config_local": config_local_snap,
        }
        bstep["status"] = "done"
        bstep["finished_at"] = _now()
        steps["build"] = bstep
        _save_run_json(exp_dir, meta)

    # 最终返回 meta
    return meta


# alias (internal) to satisfy mypy-friendly call above
partition_fn = partition


def search_experiment(
    name: str,
    global_cfg: GlobalConfig,
    dataset: DatasetSpec,
    search: SearchParams,
) -> Dict[str, Any]:
    exp_dir = os.path.join(dataset.work_dir, "experiments", name)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}. Build it first.")

    # Load & extend metadata
    run_json = os.path.join(exp_dir, "run.json")
    meta = {}
    if os.path.exists(run_json):
        with open(run_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.setdefault("steps", {})

    search_info = search_starling_auto(name, global_cfg, dataset, search, exp_dir)
    meta["steps"]["search"] = search_info

    _write_text(run_json, json.dumps(meta, indent=2))
    return {"experiment": name, **search_info}

# -----------------------------------------------------------------------------
# (Optional) helpers for inspection
# -----------------------------------------------------------------------------

def list_experiments(dataset: DatasetSpec) -> List[str]:
    base = os.path.join(dataset.work_dir, "experiments")
    if not os.path.isdir(base):
        return []
    return sorted([d.name for d in pathlib.Path(base).iterdir() if d.is_dir()])


def show_experiment(dataset: DatasetSpec, name: str) -> Dict[str, Any]:
    path = os.path.join(dataset.work_dir, "experiments", name, "run.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"run.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------

def _ensure_trailing_slash(p: str) -> str:
    return p if p.endswith("/") else (p + "/")

def _merge_overrides(base: Dict[str, str], extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    out = dict(base or {})
    if extra:
        out.update(extra)
    return out

def build_and_search_five_index_sets(
    *,
    global_cfg: GlobalConfig,
    dataset: DatasetSpec,
    generate_config_dataset: str,   # BuildParams.generate_config_dataset
    num_shards: int = 20,
    kmeans_sample_ratio: float = 0.01,
    kmeans_dups: Tuple[int, int] = (1, 2),
    name_prefix: str = "exp5new",
    force: bool = False,

    # SPANN/ELPIS 你自己填（因为它们的“分片数量”由内部配置决定）
    spann_params: Optional[SPANNParams] = None,
    elpis_params: Optional[ELPISParams] = None,

    # 搜索参数：给一个“基准 SearchParams”，每个实验都用它来搜
    base_search: Optional[SearchParams] = None,

    # 可选：对不同方法的搜索覆盖 config_local_overrides（只覆盖 env，不改其它字段）
    # keys: "natural", "kmeans_dup1", "kmeans_dup2", "spann", "elpis"
    search_overrides_by_kind: Optional[Dict[str, Dict[str, str]]] = None,

    # 可选：统一覆盖 BuildParams（比如 variants/env）
    build_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build + Search 5 index sets for the same dataset:
      1) natural partition (num_shards)
      2) kmeans partition dup=1
      3) kmeans partition dup=2
      4) spann partition (spann_params required)
      5) elpis partition (elpis_params required)

    Returns:
      {
        "natural": {"build": meta, "search": meta},
        "kmeans_dup1": {...},
        ...
      }
    """

    if base_search is None:
        # 给个尽量“安全”的默认值（你也可以自己传进来）
        base_search = SearchParams(
            build_type="release",
            mode="search_split",
            algo="knn",
            shard_id=0,
            config_local_overrides={},
        )

    results: Dict[str, Any] = {}

    # ---- shared BuildParams ----
    B = BuildParams(
        generate_config_dataset=generate_config_dataset,
        config_local_path=os.path.join(global_cfg.scripts_dir, "config_local.sh"),
        dtype=dataset.dtype,
    )
    if build_overrides:
        for k, v in build_overrides.items():
            setattr(B, k, v)

    def run_one(kind: str, exp_name: str, part_params) -> None:
        # 1) build
        print(f"Building {kind} partition...", flush=True)
        build_meta = create_and_build_experiment(
            name=exp_name,
            global_cfg=global_cfg,
            dataset=dataset,
            partition=part_params,
            build=B,
            force=force,
        )

        # 2) search (支持 per-kind 覆盖 env)
        print(f"Searching {kind} partition...", flush=True)
        per_kind_env = (search_overrides_by_kind or {}).get(kind, {})
        merged_env = _merge_overrides(base_search.config_local_overrides, per_kind_env)

        SP = dataclasses.replace(
            base_search,
            config_local_overrides=merged_env,
        )
        search_meta = search_experiment(
            name=exp_name,
            global_cfg=global_cfg,
            dataset=dataset,
            search=SP,
        )

        results[kind] = {"build": build_meta, "search": search_meta}

    # ---- 1) Natural ----
    print("Building + searching natural partition...", flush=True)
    natural_fmt = "fvecs" if dataset.dtype == "float" else "bvecs"
    natural_name = f"{name_prefix}_natural_{num_shards}shards"
    P_nat = NaturalParams(
        split_name=natural_name,
        out_dir=_ensure_trailing_slash(os.path.join(dataset.work_dir, "data_split", natural_name)),
        num_shards=num_shards,
        input_fmt=natural_fmt,
    )
    run_one("natural", natural_name, P_nat)

    # ---- 2/3) KMeans dup=1,2 ----
    print("Building + searching kmeans partitions...", flush=True)
    for dup in kmeans_dups:
        km_name = f"{name_prefix}_kmeans_{num_shards}shards_dup{dup}"
        P_km = KMeansParams(
            split_name=km_name,
            out_dir=_ensure_trailing_slash(os.path.join(dataset.work_dir, "data_split", km_name)),
            sample_ratio=kmeans_sample_ratio,
            num_shards=num_shards,
            dup_per_vec=dup,
            use_partition_data_exec=True,
        )
        run_one(f"kmeans_dup{dup}", km_name, P_km)

    # ---- 4) SPANN ----
    print("Building + searching SPANN partition...", flush=True)
    if spann_params is None:
        raise ValueError("spann_params is required for SPANN build+search.")

    P_spann = dataclasses.replace(
        spann_params,
        out_dir=_ensure_trailing_slash(spann_params.out_dir),
    )
    run_one("spann", P_spann.split_name, P_spann)

    # ---- 5) ELPIS ----
    print("Building + searching ELPIS partition...", flush=True)
    if elpis_params is None:
        raise ValueError("elpis_params is required for ELPIS build+search.")

    P_elpis = dataclasses.replace(
        elpis_params,
        out_dir=_ensure_trailing_slash(elpis_params.out_dir),
    )
    run_one("elpis", P_elpis.split_name, P_elpis)

    return results



# -----------------------------------------------------------------------------
# Example usage (not for unit test)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Global + dataset
    G = GlobalConfig(
        starling_root="/root/paper/DiskAnnPQ/starling",
        scripts_dir="/root/paper/DiskAnnPQ/starling/scripts",
        dataset_root="/root/paper",
    )

    # 2) Dataset specification
    DS_MAP = {
        "sift1m": DatasetSpec(
            name="sift1m", dim=128, dtype="float",
            base_file="/root/paper/sift1m/sift_base.fbin",
            base_file_fvecs="/root/paper/sift1m/sift_base.fvecs",  # for natural partition and elpis
            query_file="/root/paper/sift1m/sift_query.fbin",
            query_file_fvecs="/root/paper/sift1m/sift_query.fvecs",  # for spann partition
            work_dir="/root/paper/sift1m",
        ),
        "sift100m": DatasetSpec(
            name="sift100m", dim=128, dtype="uint8",
            base_file="/root/paper/sift100m/learn.100M.u8bin",
            base_file_bvecs="/root/paper/sift100m/learn.100M.bvecs",  # for natural partition
            base_file_fvecs="/root/paper/sift100m/learn.100M.fvecs",  # for elpis partition
            query_file="/root/paper/sift100m/query.public.1K.u8bin",
            query_file_fvecs="/root/paper/sift100m/query.public.1K.fvecs",  # for spann partition
            work_dir="/root/paper/sift100m",
        ),
        "sift1b": DatasetSpec(
            name="sift1b", dim=128, dtype="uint8",
            base_file="/root/paper/sift1b/bigann_base.fbin",
            query_file="/root/paper/sift1b/bigann_query.fbin",
            work_dir="/root/paper/sift1b",
        ),
    }

    DS_NAME = "sift1m"
    DS = DS_MAP[DS_NAME]


    def run_five_index_sets_sift100m():
        base_search = SearchParams(
            build_type="release",
            mode="search_split",
            algo="knn",
            shard_id=0,
            config_local_overrides={
                "MEM_USE_FREQ": "0",
                "GP_USE_FREQ": "0",
                "SPLIT_K_LIST": "(1 2 3 5 10 15 20)",
                "LS": '\"10 12 14 16 18 20 25 30 40 60\"',
                "T_LIST": "(32)",
            },
        )

        # SPANN / ELPIS 你自己填（这里随便举例）
        P_spann = SPANNParams(
            split_name="exp5_spann_custom",
            out_dir="/root/paper/sift100m/data_split/exp5_spann_custom/",
            template="default-1b",
            overrides={
                "SelectHead.BKTKmeansK": "2",
                "SelectHead.SamplesNumber": "1000",
                "SelectHead.BKTLeafSize": f"{int(1e8 // 10)}",
                "SelectHead.SplitThreshold": f"{int(1e8 // 5)}",
                "SelectHead.Ratio": f"{20 / 1e8}",
                "BuildSSDIndex.PostingPageLimit": "50000000",
            },
        )

        P_elpis = ELPISParams(
            split_name="exp5_elpis_custom_1230",
            out_dir="/root/paper/sift100m/data_split/exp5_elpis_custom_1230/",
            index_path="/root/paper/sift100m/elpis_index/exp5_elpis_custom_1230/",
            dataset_size=100000000,
            timeseries_size=128,
            leaf_size=12000000,
            kb=16,
            Lb=200,
            buffer_size=64,
            run_query=False,
        )

        res = build_and_search_five_index_sets(
            global_cfg=G,
            dataset=DS_MAP["sift100m"],
            generate_config_dataset="sift100m",
            num_shards=20,
            kmeans_sample_ratio=0.01,
            kmeans_dups=(1, 2),
            name_prefix="exp5",
            force=False,

            spann_params=P_spann,
            elpis_params=P_elpis,

            base_search=base_search,
            search_overrides_by_kind={
                "natural": {
                    "SPLIT_K_LIST": "(1)",
                },
                "elpis": {
                    "ELPIS_NPROBES": "5",
                },
            },
        )

    