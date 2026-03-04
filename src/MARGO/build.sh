cd /root/paper/DiskAnnPQ/MARGO/release && \
make -j && \

cd /root/paper/DiskAnnPQ/MARGO/scripts && \
source config_dataset.sh && \
dataset_SIFT1m && \
source config_local.sh && \
bash ./run_benchmark.sh release build && \
INDEX_PREFIX_PATH="../index/${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/" && \
echo INDEX_PREFIX_PATH: ${INDEX_PREFIX_PATH} && \
/root/paper/DiskAnnPQ/MARGO/release/my_gp/new_mincut ${INDEX_PREFIX_PATH} ${BASE_FPATH} && \
/root/paper/DiskAnnPQ/MARGO/release/tests/utils/index_relayout ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_partition.bin && \
if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then \
    mv ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_disk_beam_search.index; \
fi && \
mv ${INDEX_PREFIX_PATH}_partition_tmp.index ${INDEX_PREFIX_PATH}_disk.index && \
bash ./run_benchmark.sh release build_mem && \
bash ./run_benchmark.sh release search knn
# Notice that MARGO must be build at one shot, otherwise will segfault