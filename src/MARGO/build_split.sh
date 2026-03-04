SPLIT_NAME=natural5

cd /root/paper/DiskAnnPQ/MARGO/scripts
bash auto_generate_config_and_run.sh generate_config sift1b ${SPLIT_NAME} uint8
bash auto_generate_config_and_run.sh run_benchmarks sift1b  ${SPLIT_NAME} "release build" && \
bash auto_generate_config_and_run.sh run_benchmarks sift1b  ${SPLIT_NAME} "release build_mem" && \
bash auto_generate_config_and_run.sh run_benchmarks sift1b  ${SPLIT_NAME} "release mcrun" && \
bash ./run_benchmark_multi_split.sh release search_split knn