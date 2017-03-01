#!/bin/bash
# Usage:
# ./action_experiments/scripts/train_action_det.sh GPU NET DATASET STACKED_LEN MOD SPLIT [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./action_experiments/scripts/train_action_det.sh 0 VGG_16 UCF-Sports 1 RGB 0
# ./action_experiments/scripts/train_action_det.sh 0 VGG_16 UCF-Sports 5 RGB 0
# ./action_experiments/scripts/train_action_det.sh 0 VGG_16 JHMDB 1 RGB 0
# ./action_experiments/scripts/train_action_det.sh 0 VGG_16 UCF101 1 RGB 0
# 

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
STACKED_LEN=$4
MOD=$5
SPLIT=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB=${DATASET}_${MOD}_${STACKED_LEN}_split_${SPLIT}
ITERS=70000

LOG="action_experiments/logs/train_${TRAIN_IMDB}_${NET}_${EXTRA_ARGS_SLUG}.txt" #.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./action_tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${DATASET}/${NET}/${MOD}_${STACKED_LEN}_solver.prototxt \
  --weights models/imagenet_models/${STACKED_LEN}_${NET}.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg action_experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`

