SPLIT_NAME=natural5shard
DATASET=spacev1b

cd /root/paper/DiskAnnPQ/Gorgeous/scripts
bash auto_generate_config_and_run.sh generate_config ${DATASET} ${SPLIT_NAME} uint8
# bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release build" && \
# bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release build_mem" && \
bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release split_graph" && \
bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release gr_layout" && \
bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release search knn" && \
bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release gp" && \
bash auto_generate_config_and_run.sh run_benchmarks ${DATASET}  ${SPLIT_NAME} "release search knn"