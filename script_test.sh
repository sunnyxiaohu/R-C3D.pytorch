#!/bin/bash
set -x
NET_DIR=$1
EX_DIR=$2
FRAMERATE=25
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

export PYTHONUNBUFFERED=true

LOG="output/${NET_DIR}/${EX_DIR}/test_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python ./test_net.py \
  --net ${NET_DIR} \
  --dataset ${EX_DIR} \
  --cuda \
  ${EXTRA_ARGS}

#evaluation
python ./evaluation/${EX_DIR}/${EX_DIR}_log_analysis.py $LOG --framerate ${FRAMERATE}
