#!/bin/bash
#
# Multi-Shard DiskANN Pipeline Script
# Handles: Data Partition -> Index Build -> Multi-Shard Search
#

set -e

#=============================================================================
# Configuration - Modify these parameters as needed
#=============================================================================

# Data configuration
DATA_TYPE="float"                    # float, int8, uint8
DIST_FN="l2"                         # l2, mips, cosine
BASE_FILE=""                         # Input data file path (required)
QUERY_FILE=""                        # Query file path (required)
GT_FILE=""                           # Ground truth file path (optional, for recall)

# Partition configuration
PARTITION_TYPE="natural"             # natural, knn
NUM_SHARDS=4                         # Number of shards
OUTPUT_PREFIX=""                     # Output prefix for partitioned data (required)
# KNN partition specific
KNN_SAMPLING_RATE=0.1                # Sampling rate for knn partition
KNN_K_INDEX=1                        # Replication factor (1 = no replication)

# Index build configuration
INDEX_R=64                           # Graph degree
INDEX_L=100                          # Build L parameter
INDEX_B=0.003                        # DRAM budget / PQ bytes
INDEX_M=1                            # Memory limit in GB
BUILD_THREADS=16                     # Build threads
PQ_TYPE="PQ"                         # PQ, OPQ, LSQ, RABITQ, FLAT

# Search configuration
SEARCH_K=10                          # Top-K results
SEARCH_W=2                           # Beamwidth
SEARCH_THREADS=16                    # Search threads
CACHE_NODES=0                        # Nodes to cache
L_INIT=20                            # Initial L for multi-shard
L_BIG=20                             # L increment for contributing shards
L_SMALL=10                           # L increment for non-contributing shards
L_END=0                              # Termination L sum (0 = auto)

# Paths - auto-configured
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${DISKANN_ROOT}/build"

# Starling partition tool path (adjust if different)
STARLING_ROOT="${DISKANN_ROOT}/../starling"
PARTITION_TOOL="${STARLING_ROOT}/debug/tests/utils/run_data_partition"

#=============================================================================
# Parse command line arguments
#=============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Multi-Shard DiskANN Pipeline: Partition -> Build -> Search"
    echo ""
    echo "Required options:"
    echo "  --base_file PATH        Input data file"
    echo "  --output_prefix PATH    Output prefix for shards and indexes"
    echo "  --query_file PATH       Query file for search"
    echo ""
    echo "Data options:"
    echo "  --data_type TYPE        Data type: float, int8, uint8 (default: float)"
    echo "  --dist_fn FN            Distance function: l2, mips, cosine (default: l2)"
    echo "  --gt_file PATH          Ground truth file for recall calculation"
    echo ""
    echo "Partition options:"
    echo "  --partition_type TYPE   Partition type: natural, knn (default: natural)"
    echo "  --num_shards N          Number of shards (default: 4)"
    echo "  --knn_sampling_rate R   KNN sampling rate (default: 0.1)"
    echo "  --knn_k_index K         KNN replication factor (default: 1)"
    echo ""
    echo "Index build options:"
    echo "  --index_R R             Graph degree (default: 64)"
    echo "  --index_L L             Build L parameter (default: 100)"
    echo "  --index_B B             PQ bytes (default: 0.003)"
    echo "  --build_threads T       Build threads (default: 16)"
    echo "  --pq_type TYPE          PQ, OPQ, LSQ, RABITQ, FLAT (default: PQ)"
    echo ""
    echo "Search options:"
    echo "  --search_K K            Top-K results (default: 10)"
    echo "  --search_W W            Beamwidth (default: 4)"
    echo "  --search_threads T      Search threads (default: 16)"
    echo "  --L_init L              Initial L (default: 20)"
    echo "  --L_big L               L increment for big shards (default: 20)"
    echo "  --L_small L             L increment for small shards (default: 10)"
    echo ""
    echo "Pipeline control:"
    echo "  --skip_partition        Skip partition step"
    echo "  --skip_build            Skip index build step"
    echo "  --skip_search           Skip search step"
    echo ""
    echo "Example:"
    echo "  $0 --base_file sift_base.fvecs --output_prefix ./shards/sift \\"
    echo "     --query_file sift_query.fvecs --gt_file sift_gt.ivecs \\"
    echo "     --num_shards 4 --partition_type natural"
}

SKIP_PARTITION=0
SKIP_BUILD=0
SKIP_SEARCH=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_file) BASE_FILE="$2"; shift 2 ;;
        --output_prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --query_file) QUERY_FILE="$2"; shift 2 ;;
        --gt_file) GT_FILE="$2"; shift 2 ;;
        --data_type) DATA_TYPE="$2"; shift 2 ;;
        --dist_fn) DIST_FN="$2"; shift 2 ;;
        --partition_type) PARTITION_TYPE="$2"; shift 2 ;;
        --num_shards) NUM_SHARDS="$2"; shift 2 ;;
        --knn_sampling_rate) KNN_SAMPLING_RATE="$2"; shift 2 ;;
        --knn_k_index) KNN_K_INDEX="$2"; shift 2 ;;
        --index_R) INDEX_R="$2"; shift 2 ;;
        --index_L) INDEX_L="$2"; shift 2 ;;
        --index_B) INDEX_B="$2"; shift 2 ;;
        --build_threads) BUILD_THREADS="$2"; shift 2 ;;
        --pq_type) PQ_TYPE="$2"; shift 2 ;;
        --search_K) SEARCH_K="$2"; shift 2 ;;
        --search_W) SEARCH_W="$2"; shift 2 ;;
        --search_threads) SEARCH_THREADS="$2"; shift 2 ;;
        --L_init) L_INIT="$2"; shift 2 ;;
        --L_big) L_BIG="$2"; shift 2 ;;
        --L_small) L_SMALL="$2"; shift 2 ;;
        --skip_partition) SKIP_PARTITION=1; shift ;;
        --skip_build) SKIP_BUILD=1; shift ;;
        --skip_search) SKIP_SEARCH=1; shift ;;
        --help|-h) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$BASE_FILE" ]] || [[ -z "$OUTPUT_PREFIX" ]] || [[ -z "$QUERY_FILE" ]]; then
    echo "Error: --base_file, --output_prefix, and --query_file are required"
    print_usage
    exit 1
fi

#=============================================================================
# Helper functions
#=============================================================================

log_step() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
}

check_file_exists() {
    if [[ ! -f "$1" ]]; then
        echo "Error: File not found: $1"
        exit 1
    fi
}

get_file_format() {
    local file="$1"
    if [[ "$file" == *.fvecs ]]; then
        echo "fvecs"
    elif [[ "$file" == *.bvecs ]]; then
        echo "bvecs"
    elif [[ "$file" == *.bin ]]; then
        echo "bin"
    else
        echo "unknown"
    fi
}

#=============================================================================
# Step 1: Data Partition
#=============================================================================

run_partition() {
    log_step "Step 1: Data Partition"
    
    check_file_exists "$BASE_FILE"
    
    # Create output directory
    OUTPUT_DIR=$(dirname "$OUTPUT_PREFIX")
    mkdir -p "$OUTPUT_DIR"
    
    FILE_FORMAT=$(get_file_format "$BASE_FILE")
    
    echo "Partition type: $PARTITION_TYPE"
    echo "Input file: $BASE_FILE"
    echo "Output prefix: $OUTPUT_PREFIX"
    echo "Number of shards: $NUM_SHARDS"
    
    if [[ "$PARTITION_TYPE" == "natural" ]]; then
        if [[ ! -f "$PARTITION_TOOL" ]]; then
            echo "Error: Partition tool not found: $PARTITION_TOOL"
            echo "Please build starling first: cd starling/scripts && ./run_benchmark.sh debug"
            exit 1
        fi
        
        "$PARTITION_TOOL" natural "$BASE_FILE" "$OUTPUT_PREFIX" "$NUM_SHARDS" "$FILE_FORMAT"
        
    elif [[ "$PARTITION_TYPE" == "knn" ]]; then
        if [[ ! -f "$PARTITION_TOOL" ]]; then
            echo "Error: Partition tool not found: $PARTITION_TOOL"
            exit 1
        fi
        
        # knn args: <base_file> <output_prefix> <num_shards> <data_type> <sampling_rate> <num_partitions> <k_index>
        "$PARTITION_TOOL" knn "$BASE_FILE" "$OUTPUT_PREFIX" "$NUM_SHARDS" "$DATA_TYPE" \
            "$KNN_SAMPLING_RATE" "$NUM_SHARDS" "$KNN_K_INDEX"
    else
        echo "Error: Unknown partition type: $PARTITION_TYPE"
        exit 1
    fi
    
    echo "Partition complete. Generated files:"
    ls -la "${OUTPUT_PREFIX}"_subshard-* 2>/dev/null || true
}

#=============================================================================
# Step 2: Build Index for Each Shard
#=============================================================================

run_build() {
    log_step "Step 2: Build Disk Index for Each Shard"
    
    BUILD_DISK_INDEX="${BUILD_DIR}/apps/build_disk_index"
    if [[ ! -f "$BUILD_DISK_INDEX" ]]; then
        echo "Error: build_disk_index not found: $BUILD_DISK_INDEX"
        echo "Please build DiskANN first: mkdir build && cd build && cmake .. && make -j"
        exit 1
    fi
    
    for ((i=0; i<NUM_SHARDS; i++)); do
        SHARD_DATA="${OUTPUT_PREFIX}_subshard-${i}.bin"
        SHARD_INDEX="${OUTPUT_PREFIX}_subshard-${i}_index"
        SHARD_IDMAP="${OUTPUT_PREFIX}_subshard-${i}_ids_uint32.bin"
        
        if [[ ! -f "$SHARD_DATA" ]]; then
            echo "Warning: Shard data not found: $SHARD_DATA, skipping..."
            continue
        fi
        
        echo ""
        echo "Building index for shard $i..."
        echo "  Data: $SHARD_DATA"
        echo "  Index: $SHARD_INDEX"
        
        "$BUILD_DISK_INDEX" \
            --data_type "$DATA_TYPE" \
            --dist_fn "$DIST_FN" \
            --data_path "$SHARD_DATA" \
            --index_path_prefix "$SHARD_INDEX" \
            -R "$INDEX_R" \
            -L "$INDEX_L" \
            -B "$INDEX_B" \
            -M "$INDEX_M" \
            -T "$BUILD_THREADS"
        
        # Copy ID map to match index prefix for search
        INDEX_IDMAP="${SHARD_INDEX}_ids_uint32.bin"
        if [[ -f "$SHARD_IDMAP" ]] && [[ ! -f "$INDEX_IDMAP" ]]; then
            echo "  Copying ID map: $SHARD_IDMAP -> $INDEX_IDMAP"
            cp "$SHARD_IDMAP" "$INDEX_IDMAP"
        fi
    done
    
    echo ""
    echo "Index build complete."
}

#=============================================================================
# Step 3: Multi-Shard Search
#=============================================================================

run_search() {
    log_step "Step 3: Multi-Shard Search"
    
    SEARCH_MULTI_SHARD="${BUILD_DIR}/apps/search_multi_shard_index"
    if [[ ! -f "$SEARCH_MULTI_SHARD" ]]; then
        echo "Error: search_multi_shard_index not found: $SEARCH_MULTI_SHARD"
        exit 1
    fi
    
    check_file_exists "$QUERY_FILE"
    
    # Build index path list
    INDEX_PATHS=""
    for ((i=0; i<NUM_SHARDS; i++)); do
        SHARD_INDEX="${OUTPUT_PREFIX}_subshard-${i}_index"
        if [[ -n "$INDEX_PATHS" ]]; then
            INDEX_PATHS="${INDEX_PATHS},"
        fi
        INDEX_PATHS="${INDEX_PATHS}${SHARD_INDEX}"
    done
    
    RESULT_PATH="${OUTPUT_PREFIX}_search_results"
    
    echo "Index paths: $INDEX_PATHS"
    echo "Query file: $QUERY_FILE"
    echo "Result path: $RESULT_PATH"
    
    GT_OPTION=""
    if [[ -n "$GT_FILE" ]] && [[ -f "$GT_FILE" ]]; then
        GT_OPTION="--gt_file $GT_FILE"
        echo "Ground truth: $GT_FILE"
    fi
    
    "$SEARCH_MULTI_SHARD" \
        --data_type "$DATA_TYPE" \
        --dist_fn "$DIST_FN" \
        --index_path_prefixes "$INDEX_PATHS" \
        --query_file "$QUERY_FILE" \
        --result_path "$RESULT_PATH" \
        -K "$SEARCH_K" \
        -W "$SEARCH_W" \
        -T "$SEARCH_THREADS" \
        -Q "$PQ_TYPE" \
        --L_init "$L_INIT" \
        --L_big "$L_BIG" \
        --L_small "$L_SMALL" \
        --L_end "$L_END" \
        $GT_OPTION
}

#=============================================================================
# Main Pipeline
#=============================================================================

echo "============================================================"
echo "  Multi-Shard DiskANN Pipeline"
echo "============================================================"
echo "Configuration:"
echo "  Base file: $BASE_FILE"
echo "  Output prefix: $OUTPUT_PREFIX"
echo "  Query file: $QUERY_FILE"
echo "  Data type: $DATA_TYPE"
echo "  Distance: $DIST_FN"
echo "  Partition: $PARTITION_TYPE ($NUM_SHARDS shards)"
echo "  Index: R=$INDEX_R, L=$INDEX_L, B=$INDEX_B"
echo "  Search: K=$SEARCH_K, W=$SEARCH_W"

if [[ $SKIP_PARTITION -eq 0 ]]; then
    run_partition
else
    echo ""
    echo "Skipping partition step..."
fi

if [[ $SKIP_BUILD -eq 0 ]]; then
    run_build
else
    echo ""
    echo "Skipping build step..."
fi

if [[ $SKIP_SEARCH -eq 0 ]]; then
    run_search
else
    echo ""
    echo "Skipping search step..."
fi

log_step "Pipeline Complete"
echo "Results saved to: ${OUTPUT_PREFIX}_search_results_*"
