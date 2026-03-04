#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

# your path

dataset_t2i100m() {
  BASE_PATH=/root/paper/text2image1b/base.100M.fbin
  QUERY_FILE=/root/paper/text2image1b/query.10K.fbin
  GT_FILE=/root/paper/text2image1b/query.10K.fbin.base100m.k100.gt
  PREFIX=t2i100m
  DATA_TYPE=float
  DIST_FN=mips
  K=10
  DATA_DIM=100
  DATA_N=100000000
  SECTOR_LEN=4096
  GR_SECTOR_LEN=4096
  N_PQ_CODE=4
}
