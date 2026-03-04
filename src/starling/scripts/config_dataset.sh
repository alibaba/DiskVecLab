#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

dataset_bigann1M() {
  BASE_PATH=/root/paper/sift1m/data_split/knn/_subshard-1.bin
  QUERY_FILE=/root/paper/sift1m/sift_query.fbin
  GT_FILE=/root/paper/sift1m/sift_starling_k10_gt
  PREFIX=bigann_1m
  DATA_TYPE=float
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=485820
}

