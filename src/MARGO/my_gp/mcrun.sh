cd ../release && make -j && \

# SIFT100K
path_prefix="../index/sift100k_M64_R64_L125_B0.003/"
base_file="../data/sift100k/sift_learn.fbin"

./my_gp/new_mincut ${path_prefix} ${base_file}

./tests/utils/index_relayout ${path_prefix}_disk.index ${path_prefix}_partition.bin
if [ ! -f "${path_prefix}_disk_beam_search.index" ]; then
    mv ${path_prefix}_disk.index ${path_prefix}_disk_beam_search.index
fi
mv ${path_prefix}_partition_tmp.index ${path_prefix}_disk.index