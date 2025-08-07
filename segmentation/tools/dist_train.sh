#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup torchrun $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} > rmt_t_70e_segmentation_running.log 2>&1 &

# python -m torch.distributed.launch --nproc_per_node=2 $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
