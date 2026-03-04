#!/bin/bash
set -e

# Automated script to generate config_dataset_auto.sh and run run_benchmark.sh
# Usage: ./auto_generate_config_and_run.sh <action> <dataset_name> <split_type> <input_params>

# Check parameters
if [ $# -lt 3 ]; then
    echo "Usage: $0 <action> <dataset_name> <split_type> [input_params]"
    echo "Actions: generate_config, run_benchmarks"
    echo "Example: $0 generate_config sift1m knn10 float"
    echo "Example: $0 run_benchmarks sift1m knn10 \"debug build knn\""
    exit 1
fi

ACTION=$1
DATASET_NAME=$2
SPLIT_TYPE=$3
# float/uint8 for generate_config
INPUT_PARAMS=$4

# Fixed path prefix, using DATASET_NAME instead of sift1m
BASE_PATH_PREFIX="/root/paper/${DATASET_NAME}/data_split"

# Tool script path (relative path)
INSPECT_SCRIPT="../../tools/python/inspect_data.py"

# Generated config file path (relative path)
CONFIG_FILE="config_dataset_auto.sh"

# Generate config_dataset_auto.sh file
generate_config() {
    echo "#!/bin/sh" > $CONFIG_FILE
    echo "" >> $CONFIG_FILE
    echo "# Auto-generated config_dataset_auto.sh" >> $CONFIG_FILE
    echo "# Dataset name: $DATASET_NAME" >> $CONFIG_FILE
    echo "# Split type: $SPLIT_TYPE" >> $CONFIG_FILE
    echo "" >> $CONFIG_FILE
    
    # Iterate through only _subshard-{number}.bin files (excluding _ids_uint32 variants)
    for file in $BASE_PATH_PREFIX/$SPLIT_TYPE/_subshard-[0-9]*.bin; do
        # Output an error if no files found and exit
        if [ ! -e "$file" ]; then
            echo "Error: No files found matching pattern $BASE_PATH_PREFIX/$SPLIT_TYPE/_subshard-[0-9]*.bin"
            exit 1
        fi
        echo "Processing file: $file"

        # Skip files that match the _ids_uint32 pattern
        if [[ $(basename "$file") == *_ids_uint32.bin ]]; then
            continue
        fi
        
        # Get filename and shard_id
        filename=$(basename "$file")
        shard_id=$(echo "$filename" | sed -E 's/_subshard-([0-9]+)\.bin/\1/')
        
        # Get dimension and row count of data file
        echo "Inspecting data file to get dimension and row count..."
        echo "Command: python3 $INSPECT_SCRIPT \"$file\" ${INPUT_PARAMS} 10"
        output=$(python3 $INSPECT_SCRIPT "$file" ${INPUT_PARAMS} 10 2>&1)
        DATA_N=$(echo "$output" | grep "Total points" | awk '{print $3}' | sed 's/,//')
        DATA_DIM=$(echo "$output" | grep "Dim:" | awk '{print $5}')
        
        # Generate function name
        echo "Generating function for shard_id: $shard_id, Data N: $DATA_N, Data Dim: $DATA_DIM"
        func_name="dataset_${DATASET_NAME}_${SPLIT_TYPE}_shard_${shard_id}"
        
        # Write function definition
        echo "$func_name() {" >> $CONFIG_FILE
        echo "  export BASE_PATH=$file" >> $CONFIG_FILE
        echo "  export QUERY_FILE=/root/paper/${DATASET_NAME}/${DATASET_NAME}_query.fbin" >> $CONFIG_FILE
        echo "  export GT_FILE=/root/paper/${DATASET_NAME}/${DATASET_NAME}_starling_k10_gt" >> $CONFIG_FILE
        echo "  export PREFIX=${DATASET_NAME}_${SPLIT_TYPE}_shard_${shard_id}" >> $CONFIG_FILE
        echo "  export DATA_TYPE=${INPUT_PARAMS}" >> $CONFIG_FILE
        echo "  export DIST_FN=l2" >> $CONFIG_FILE
        echo "  export K=10" >> $CONFIG_FILE
        echo "  export DATA_DIM=$DATA_DIM" >> $CONFIG_FILE
        echo "  export DATA_N=$DATA_N" >> $CONFIG_FILE
        echo "  export DATA_PATH=$BASE_PATH_PREFIX/$SPLIT_TYPE" >> $CONFIG_FILE
        echo "  export SPLIT_TYPE=$SPLIT_TYPE" >> $CONFIG_FILE
        echo "}" >> $CONFIG_FILE
        echo "" >> $CONFIG_FILE
    done
    
    echo "Generated $CONFIG_FILE successfully!"
}

# Run benchmarks
run_benchmarks() {
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file $CONFIG_FILE does not exist. Please run generate_config first."
        exit 1
    fi
    
    # Get all shard_ids and sort them in ascending order
    shard_ids=()
    for file in $BASE_PATH_PREFIX/$SPLIT_TYPE/_subshard-[0-9]*.bin; do
        # Skip files that match the _ids_uint32 pattern
        if [[ $(basename "$file") == *_ids_uint32.bin ]]; then
            continue
        fi
        
        filename=$(basename "$file")
        shard_id=$(echo "$filename" | sed -E 's/_subshard-([0-9]+)\.bin/\1/')
        shard_ids+=($shard_id)
    done
    
    # Sort shard_ids in ascending order
    IFS=$'\n' shard_ids=($(sort -n <<<"${shard_ids[*]}"))
    unset IFS

    # Run benchmark for each shard sequentially
    for shard_id in "${shard_ids[@]}"; do
        func_name="dataset_${DATASET_NAME}_${SPLIT_TYPE}_shard_${shard_id}"
        echo "Running benchmark for $func_name..."

        # Source config_dataset_auto.sh and the corresponding function directly
        source $CONFIG_FILE
        $func_name

        # Run benchmark
        ./run_benchmark.sh $INPUT_PARAMS
    done
    
    echo "All benchmarks completed!"
}

# Main flow
main() {
    case $ACTION in
        generate_config)
            echo "Generating config for dataset: $DATASET_NAME, split type: $SPLIT_TYPE"
            generate_config
            ;;
        run_benchmarks)
            echo "Running benchmarks for dataset: $DATASET_NAME, split type: $SPLIT_TYPE with params: $INPUT_PARAMS"
            run_benchmarks
            ;;
        *)
            echo "Invalid action: $ACTION"
            echo "Valid actions: generate_config, run_benchmarks"
            exit 1
            ;;
    esac
}

# Execute main flow
main
