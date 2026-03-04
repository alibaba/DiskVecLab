#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

#################
#    SIFT100K   #
#################
dataset_SIFT100K() {
  BASE_PATH=../data/sift100k/sift_learn.fbin
  QUERY_FILE=../data/sift100k/sift_query.fbin
  GT_FILE=../data/sift100k/sift_query_learn_gt100
  PREFIX=sift100k
  DATA_TYPE=float
  DIST_FN=l2
  B=0.003
  K=10
  DATA_DIM=128
  DATA_N=100000
}
