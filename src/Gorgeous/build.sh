

# 1. Gorgeous
cd /root/paper/DiskAnnPQ/Gorgeous/scripts
source config_dataset.sh && \
dataset_laion50m
echo "" >> /root/paper/DiskAnnPQ/Gorgeous/scripts/config_local.sh && \
echo "DECO_IMPL=1                     # 1 to enable Gorgeous" >> /root/paper/DiskAnnPQ/Gorgeous/scripts/config_local.sh && \
bash run_benchmark.sh release build
bash run_benchmark.sh release build_mem
bash run_benchmark.sh release split_graph
bash run_benchmark.sh release gr_layout
bash run_benchmark.sh release search knn

# 2. Starling
cd /root/paper/DiskAnnPQ/Gorgeous/scripts
source config_dataset.sh && \
echo "" >> /root/paper/DiskAnnPQ/Gorgeous/scripts/config_local.sh && \
echo "DECO_IMPL=0                     # 1 to enable Gorgeous" >> /root/paper/DiskAnnPQ/Gorgeous/scripts/config_local.sh && \
bash run_benchmark.sh release gp && \
bash run_benchmark.sh release search knn