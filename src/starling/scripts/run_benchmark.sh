#!/bin/bash

set -e
# set -x

source config_local.sh

# Allow overriding bash array params (space-separated) from environment.
# Example: BM_LIST_OVERRIDE="2 4 8".
if [ -n "${BM_LIST_OVERRIDE:-}" ]; then
  read -r -a BM_LIST <<< "${BM_LIST_OVERRIDE}"
fi
if [ -n "${T_LIST_OVERRIDE:-}" ]; then
  read -r -a T_LIST <<< "${T_LIST_OVERRIDE}"
fi
if [ -n "${SPLIT_K_LIST_OVERRIDE:-}" ]; then
  read -r -a SPLIT_K_LIST <<< "${SPLIT_K_LIST_OVERRIDE}"
fi

# Allow ad-hoc overrides from the environment for quick experimentation without
# editing config_local.sh. These are no-ops unless the *_OVERRIDE variables are set.
USE_ADAPTIVE_SEARCH="${USE_ADAPTIVE_SEARCH_OVERRIDE:-${USE_ADAPTIVE_SEARCH:-0}}"
ADAPTIVE_SHARD_SCHEDULER="${ADAPTIVE_SHARD_SCHEDULER_OVERRIDE:-${ADAPTIVE_SHARD_SCHEDULER:-${adaptive_shard_scheduler:-contrib}}}"
L_INIT="${L_INIT_OVERRIDE:-${L_INIT:-20}}"
L_BIG="${L_BIG_OVERRIDE:-${L_BIG:-20}}"
L_SMALL="${L_SMALL_OVERRIDE:-${L_SMALL:-10}}"
L_END="${L_END_OVERRIDE:-${L_END:-0}}"
DEBUG_ADAPTIVE="${DEBUG_ADAPTIVE_OVERRIDE:-${DEBUG_ADAPTIVE:-0}}"
LS="${LS_OVERRIDE:-${LS}}"
SKIP_BUILD="${SKIP_BUILD_OVERRIDE:-${SKIP_BUILD:-0}}"

INDEX_PREFIX_PATH="${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"
MEM_SAMPLE_PATH="${INDEX_PREFIX_PATH}SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
MEM_INDEX_PATH="${INDEX_PREFIX_PATH}MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}_MEM_USE_FREQ${MEM_USE_FREQ}_RANDOM_RATE${MEM_RAND_SAMPLING_RATE}_FREQ_RATE${MEM_FREQ_USE_RATE}/"
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_GP_USE_FREQ${GP_USE_FREQ}_CUT${GP_CUT}/"
FREQ_PATH="${INDEX_PREFIX_PATH}FREQ/NQ_${FREQ_QUERY_CNT}_BM_${FREQ_BM}_L_${FREQ_L}_T_${FREQ_T}/"

SUMMARY_FILE_PATH="../indices/summary.log"

extract_adaptive_params_line() {
  # Try to extract the single-line adaptive-search parameters from raw logs.
  # Example:
  #   L_capacity=15 L_init=16 L_big=10 L_small=2 L_end=90
  # If not present (non-adaptive logs), prints nothing.
  local log_file="$1"
  grep -m1 -E "L_capacity=[0-9]+.*L_init=[0-9]+.*L_big=[0-9]+.*L_small=[0-9]+.*L_end=[0-9]+" "$log_file" 2>/dev/null || true
}

parse_adaptive_params() {
  # Input: a single line like "L_capacity=15 L_init=16 L_big=10 L_small=2 L_end=90"
  # Output: "15 16 10 2 90" (or empty if parse fails)
  local line="$1"
  awk -v s="$line" '
    BEGIN {
      n = split(s, f, /[[:space:]]+/);
      for (i = 1; i <= n; i++) {
        split(f[i], kv, "=");
        if (kv[1] == "L_capacity") cap = kv[2];
        else if (kv[1] == "L_init") init = kv[2];
        else if (kv[1] == "L_big") big = kv[2];
        else if (kv[1] == "L_small") small = kv[2];
        else if (kv[1] == "L_end") end = kv[2];
      }
      if (cap != "" && init != "" && big != "" && small != "" && end != "") {
        printf "%s %s %s %s %s", cap, init, big, small, end;
      }
    }
  '
}

print_usage_and_exit() {
  echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/freq/gp/search/search_split] [knn/range]"
  exit 1
}

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

case $1 in
  debug)
    BUILD_DIR=../debug
    if [ "$SKIP_BUILD" -ne 1 ]; then
      cmake -DCMAKE_BUILD_TYPE=Debug .. -B ../debug
    fi
    EXE_PATH=$BUILD_DIR
  ;;
  release)
    BUILD_DIR=../release
    if [ "$SKIP_BUILD" -ne 1 ]; then
      cmake -DCMAKE_BUILD_TYPE=Release .. -B ../release
    fi
    EXE_PATH=$BUILD_DIR
  ;;
  *)
    print_usage_and_exit
  ;;
esac

if [ "$SKIP_BUILD" -ne 1 ]; then
  pushd $EXE_PATH
  make -j
  popd
else
  if [ ! -d "$EXE_PATH" ]; then
    echo "Build directory $EXE_PATH not found. Run without SKIP_BUILD_OVERRIDE first."
    exit 1
  fi
fi

mkdir -p ../indices && cd ../indices
PARENT_INDEX_PATH=`pwd`

date
case $2 in
  build)
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    echo "Building disk index..."
    time ${EXE_PATH}/tests/build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      -R $R \
      -L $BUILD_L \
      -B $B \
      -M $M \
      -T $BUILD_T > ${INDEX_PREFIX_PATH}build.log
    cp ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_disk_beam_search.index
  ;;
  sq)
    cp  ${INDEX_PREFIX_PATH}_disk_beam_search.index ${INDEX_PREFIX_PATH}_disk.index 
    time ${EXE_PATH}/tests/utils/sq ${INDEX_PREFIX_PATH} > ${INDEX_PREFIX_PATH}sq.log
  ;;
  build_mem)
    if [ ${MEM_USE_FREQ} -eq 1 ]; then
      if [ ! -d ${FREQ_PATH} ]; then
        echo "Seems you have not gen the freq file, run this script again: ./run_benchmark.sh [debug/release] freq [knn/range]"
        exit 1;
      fi
      echo "Parsing freq file..."

      if [ -z "${DATA_N}" ] || [ "${DATA_N}" -le 0 ]; then
        echo "ERROR: DATA_N must be > 0, current DATA_N='${DATA_N}'" >&2
        exit 1
      fi

      # 计算实际使用的采样率：保证 floor(DATA_N * rate) >= 1
      EFFECTIVE_MEM_RAND_SAMPLING_RATE=$(
        awk -v n="${DATA_N}" -v r="${MEM_RAND_SAMPLING_RATE}" '
        BEGIN {
          if (r <= 0) {
            # 如果原始采样率异常，就退化成刚好 1 个点
            r = 1.0 / n;
          }
          minr = 1.0 / n;     # 至少保证采样点数 floor(n * r) >= 1
          eff = r;
          if (eff < minr) eff = minr;
          if (eff > 1.0) eff = 1.0;  # 不要超过 1
          printf "%.10f", eff;
        }')

      echo "DATA_N=${DATA_N}, base_rate=${MEM_RAND_SAMPLING_RATE}, effective_rate=${EFFECTIVE_MEM_RAND_SAMPLING_RATE}"



      time ${EXE_PATH}/tests/utils/parse_freq_file ${DATA_TYPE} ${BASE_PATH} ${FREQ_PATH}_freq.bin ${FREQ_PATH} ${EFFECTIVE_MEM_RAND_SAMPLING_RATE} 
      MEM_DATA_PATH=${FREQ_PATH}
    else
      mkdir -p "${MEM_SAMPLE_PATH}"
      echo "Generating random slice..."

      if [ -z "${DATA_N}" ] || [ "${DATA_N}" -le 0 ]; then
        echo "ERROR: DATA_N must be > 0, current DATA_N='${DATA_N}'" >&2
        exit 1
      fi

      # 计算实际使用的采样率：保证 floor(DATA_N * rate) >= 1
      EFFECTIVE_MEM_RAND_SAMPLING_RATE=$(
        awk -v n="${DATA_N}" -v r="${MEM_RAND_SAMPLING_RATE}" '
        BEGIN {
          if (r <= 0) {
            # 如果原始采样率异常，就退化成刚好 1 个点
            r = 1.0 / n;
          }
          minr = 1.0 / n;     # 至少保证采样点数 floor(n * r) >= 1
          eff = r;
          if (eff < minr) eff = minr;
          if (eff > 1.0) eff = 1.0;  # 不要超过 1
          printf "%.10f", eff;
        }')

      echo "DATA_N=${DATA_N}, base_rate=${MEM_RAND_SAMPLING_RATE}, effective_rate=${EFFECTIVE_MEM_RAND_SAMPLING_RATE}"

      time "${EXE_PATH}/tests/utils/gen_random_slice" \
          "${DATA_TYPE}" "${BASE_PATH}" "${MEM_SAMPLE_PATH}" "${EFFECTIVE_MEM_RAND_SAMPLING_RATE}" \
          > "${MEM_SAMPLE_PATH}sample.log"

      MEM_DATA_PATH="${MEM_SAMPLE_PATH}"
    fi
    echo "Building memory index..."
    check_dir_and_make_if_absent ${MEM_INDEX_PATH}
    time ${EXE_PATH}/tests/build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_DATA_PATH} \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  freq)
    check_dir_and_make_if_absent ${FREQ_PATH}
    FREQ_LOG="${FREQ_PATH}freq.log"

    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f $DISK_FILE_PATH ]; then
      DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    fi

    echo "Generating frequency file... ${FREQ_LOG}"
    time ${EXE_PATH}/tests/search_disk_index_save_freq \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --freq_save_path $FREQ_PATH \
              --query_file $FREQ_QUERY_FILE \
              --expected_query_num $FREQ_QUERY_CNT \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${FREQ_PATH}result \
              --num_nodes_to_cache ${FREQ_CACHE} \
              -T $FREQ_T \
              -L $FREQ_L \
              -W $FREQ_BM \
              --mem_L ${FREQ_MEM_L} \
              --use_page_search 0 \
              --disk_file_path ${DISK_FILE_PATH} > ${FREQ_LOG}
  ;;
  gp)
    check_dir_and_make_if_absent ${GP_PATH}
    OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f "$OLD_INDEX_FILE" ]; then
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
    fi
    #using sq index file to gp
    GP_DATA_TYPE=$DATA_TYPE
    if [ $USE_SQ -eq 1 ]; then 
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      GP_DATA_TYPE=uint8
    fi
    GP_FILE_PATH=${GP_PATH}_part.bin
    echo "Running graph partition... ${GP_FILE_PATH}.log"
    if [ ${GP_USE_FREQ} -eq 1 ]; then
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES --freq_file ${FREQ_PATH}_freq.bin --lock_nums ${GP_LOCK_NUMS} --cut ${GP_CUT} > ${GP_FILE_PATH}.log
    else
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES > ${GP_FILE_PATH}.log
    fi

    echo "Running relayout... ${GP_PATH}relayout.log"
    time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_FILE_PATH} > ${GP_PATH}relayout.log
    if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
      mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
    fi
    #TODO: Use only one index file
    cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
    cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
  ;;
  search)
    mkdir -p ${INDEX_PREFIX_PATH}/search
    mkdir -p ${INDEX_PREFIX_PATH}/result
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # choose the disk index file by settings
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    if [ $USE_PAGE_SEARCH -eq 1 ]; then
      if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
        echo "Partition file not found. Run the script with gp option first."
        exit 1
      fi
      echo "Using Page Search"
    else
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ -f ${OLD_INDEX_FILE} ]; then
        DISK_FILE_PATH=$OLD_INDEX_FILE
      else
        echo "make sure you have not gp the index file"
      fi
      echo "Using Beam Search"
    fi

    log_arr=()
    case $3 in
      knn)
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            # Use a filesystem-safe timestamp (no spaces/colons) to avoid bash
            # redirection errors like "ambiguous redirect".
            CUR_TIME=$(date +"%Y-%m-%dT%H-%M-%S")
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_SQ${USE_SQ}_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_MEM_USE_FREQ${MEM_USE_FREQ}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_USE_FREQ${GP_USE_FREQ}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}_${CUR_TIME}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/search_disk_index --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result/result \
              --num_nodes_to_cache $CACHE \
              --quantification_type $PQ_TYPE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --disk_file_path ${DISK_FILE_PATH} \
              --use_sq ${USE_SQ}       > "${SEARCH_LOG}" 
            log_arr+=( "${SEARCH_LOG}" )
          done
        done
      ;;
      range)
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_RADIUS${RADIUS}_CACHE${CACHE}_BW${BW}_T${T}_PS${USE_PAGE_SEARCH}_PS_RATIO${PS_USE_RATIO}_ITER_KNN${RS_ITER_KNN_TO_RANGE_SEARCH}_MEM_L${MEM_L}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/range_search_disk_index \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -W $BW \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              --range_threshold $RADIUS \
              -L $RS_LS \
              --disk_file_path ${DISK_FILE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --iter_knn_to_range_search ${RS_ITER_KNN_TO_RANGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_L ${MEM_L} \
              --custom_round_num ${RS_CUSTOM_ROUND} \
              --kicked_size ${KICKED_SIZE} \
              > "${SEARCH_LOG}"
            log_arr+=( "${SEARCH_LOG}" )
          done
        done
      ;;
      *)
        print_usage_and_exit
      ;;
    esac
    if [ ${#log_arr[@]} -ge 1 ]; then
      TITLES=$(cat "${log_arr[0]}" | grep -m1 -E "^\s+L\s+")
      for f in "${log_arr[@]}"
      do
        printf "$f\n" | tee -a $SUMMARY_FILE_PATH

        ADAPTIVE_PARAMS_LINE=$(extract_adaptive_params_line "$f")
        ADAPTIVE_PARAMS_VALUES=$(parse_adaptive_params "$ADAPTIVE_PARAMS_LINE")
        if [ -n "$ADAPTIVE_PARAMS_VALUES" ]; then
          printf "%s  L_capacity  L_init  L_big  L_small  L_end\n" "${TITLES}" | tee -a $SUMMARY_FILE_PATH
          L_CAPACITY=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $1}')
          L_INIT=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $2}')
          L_BIG=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $3}')
          L_SMALL=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $4}')
          L_END=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $5}')
          cat "$f" | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | awk -v cap="$L_CAPACITY" -v init="$L_INIT" -v big="$L_BIG" -v small="$L_SMALL" -v end="$L_END" '{print $0, cap, init, big, small, end}' | tee -a $SUMMARY_FILE_PATH
        else
          printf "${TITLES}\n" | tee -a $SUMMARY_FILE_PATH
          cat "$f" | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | tee -a $SUMMARY_FILE_PATH
        fi
        printf "\n\n" >> $SUMMARY_FILE_PATH
      done
    fi
  ;;
  search_split)
    mkdir -p ${INDEX_PREFIX_PATH}/search
    mkdir -p ${INDEX_PREFIX_PATH}/result
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # choose the disk index file by settings
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    DISK_FILE_NAME=_disk.index
    if [ $USE_PAGE_SEARCH -eq 1 ]; then
      if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
        echo "Partition file not found. Run the script with gp option first."
        exit 1
      fi
      echo "Using Page Search"
    else
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ -f ${OLD_INDEX_FILE} ]; then
        DISK_FILE_PATH=$OLD_INDEX_FILE
        DISK_FILE_NAME=_disk_beam_search.index
      else
        echo "make sure you have not gp the index file"
      fi
      echo "Using Beam Search"
    fi

    log_arr=()
    case $3 in
      knn)
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            for SPLIT_K in ${SPLIT_K_LIST[@]}
            do
              # Use a filesystem-safe timestamp (no spaces/colons) to avoid bash
              # redirection errors like "ambiguous redirect".
              CUR_TIME=$(date +"%Y-%m-%dT%H-%M-%S")
              ADAPT_TAG=_ADAPT${USE_ADAPTIVE_SEARCH}_SCHED${ADAPTIVE_SHARD_SCHEDULER}_LEND${L_END}_LINIT${L_INIT}_LBIG${L_BIG}_LSMALL${L_SMALL}
              SEARCH_LOG=${PARENT_INDEX_PATH}/search_${SPLIT_TYPE}_${SPLIT_K}_SQ${USE_SQ}_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_MEM_USE_FREQ${MEM_USE_FREQ}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_USE_FREQ${GP_USE_FREQ}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}${ADAPT_TAG}_${CUR_TIME}.log
              echo "Searching... log file: ${SEARCH_LOG}"
              sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/search_disk_index_multi_split --data_type $DATA_TYPE \
                --dist_fn $DIST_FN \
                --parent_index_path $PARENT_INDEX_PATH \
                --data_path $DATA_PATH \
                --quantification_type $PQ_TYPE \
                --split_type $SPLIT_TYPE \
                --split_K $SPLIT_K \
                --disk_file_name ${DISK_FILE_NAME} \
                --query_file $QUERY_FILE \
                --gt_file $GT_FILE \
                -K $K \
                --num_nodes_to_cache $CACHE \
                -T $T \
                -L ${LS} \
                -W $BW \
                --mem_L ${MEM_L} \
                --elpis_index_path ${ELPIS_INDEX_PATH} \
                --elpis_ef ${ELPIS_EF} \
                --elpis_nworker ${ELPIS_NWORKER} \
                --elpis_nprobes ${ELPIS_NPROBES} \
                --elpis_flatt ${ELPIS_FLATT} \
                --elpis_parallel ${ELPIS_PARALLEL} \
                --use_page_search ${USE_PAGE_SEARCH} \
                --use_ratio ${PS_USE_RATIO} \
                --use_sq ${USE_SQ} \
                --use_adaptive_search ${USE_ADAPTIVE_SEARCH:-0} \
                --adaptive_shard_scheduler ${ADAPTIVE_SHARD_SCHEDULER} \
                --L_capacity 0 \
                --L_init ${L_INIT:-20} \
                --L_big ${L_BIG:-20} \
                --L_small ${L_SMALL:-10} \
                --L_end ${L_END:-0} \
                --debug ${DEBUG_ADAPTIVE:-0}       > "${SEARCH_LOG}"
              log_arr+=( "${SEARCH_LOG}" )
            done
          done
        done
      ;;
      *)
        print_usage_and_exit
      ;;
    esac
    if [ ${#log_arr[@]} -ge 1 ]; then
      TITLES=$(cat "${log_arr[0]}" | grep -m1 -E "^\s+L\s+")
      for f in "${log_arr[@]}"
      do
        printf "$f\n" | tee -a $SUMMARY_FILE_PATH

        ADAPTIVE_PARAMS_LINE=$(extract_adaptive_params_line "$f")
        ADAPTIVE_PARAMS_VALUES=$(parse_adaptive_params "$ADAPTIVE_PARAMS_LINE")
        if [ -n "$ADAPTIVE_PARAMS_VALUES" ]; then
          printf "%s  L_capacity  L_init  L_big  L_small  L_end\n" "${TITLES}" | tee -a $SUMMARY_FILE_PATH
          L_CAPACITY=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $1}')
          L_INIT=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $2}')
          L_BIG=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $3}')
          L_SMALL=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $4}')
          L_END=$(echo "$ADAPTIVE_PARAMS_VALUES" | awk '{print $5}')
          cat "$f" | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | awk -v cap="$L_CAPACITY" -v init="$L_INIT" -v big="$L_BIG" -v small="$L_SMALL" -v end="$L_END" '{print $0, cap, init, big, small, end}' | tee -a $SUMMARY_FILE_PATH
        else
          printf "${TITLES}\n" | tee -a $SUMMARY_FILE_PATH
          cat "$f" | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | tee -a $SUMMARY_FILE_PATH
        fi
        printf "\n\n" >> $SUMMARY_FILE_PATH
      done
    fi
  ;;
  *)
    print_usage_and_exit
  ;;
esac
