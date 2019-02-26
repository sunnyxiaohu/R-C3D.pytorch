#!/bin/bash
set -x
NET_DIR=$1
DATASET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

export PYTHONUNBUFFERED=true
#checkpoint 30103 checkepoch 3 --r true --lr_decay_step 10 --epochs 12
LOG="output/${NET_DIR}/${DATASET}/train_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python ./trainval_net.py \
  --net ${NET_DIR} \
  --dataset ${DATASET} \
  ${EXTRA_ARGS}
  
#2>&1 | tee $LOG

