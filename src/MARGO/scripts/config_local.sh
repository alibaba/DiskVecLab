#!/bin/sh
source config_dataset.sh

# Choose the dataset by uncomment the line below
# If multiple lines are uncommented, only the last dataset is effective
# dataset_SIFT1m

##################
#   Disk Build   #
##################
R=128
BUILD_L=128
M=400
BUILD_T=128

##################
#       SQ       #
##################
USE_SQ=0


##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=24     # graph degree
MEM_BUILD_L=128 # build complexity
MEM_ALPHA=1.2 # alpha
MEM_RAND_SAMPLING_RATE=0.001    # we use 5% for building in-memory navigation graph


#######################
#   Graph Partition   #
#######################
GP_TIMES=16
GP_T=128
GP_LOCK_NUMS=0 # will lock nodes at init, the lock_node_nums = partition_size * GP_LOCK_NUMS
GP_USE_FREQ=0 # use freq file to partition graph
GP_CUT=4096 # the graph's degree will been limited at 4096


##############
#   Search   #
##############
BM_LIST=(4)
T_LIST=(32)

CACHE=0
MEM_L=10 # non-zero to enable
# Page Search
USE_PAGE_SEARCH=1 # Set 0 for beam search, 1 for page search (default)
PS_USE_RATIO=1.0

# KNN
LS="10 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300"
