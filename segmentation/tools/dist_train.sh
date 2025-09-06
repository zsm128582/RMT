#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NAME=$3
# PORT=${PORT:-28000}
PORT=28000

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup torchrun --nproc_per_node=$GPUS --nnodes=1 --node_rank=0  --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} > $NAME.log 2>&1 &
# nohup torchrun $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} > $NAME.log 2>&1 &

# python -m torch.distributed.launch --nproc_per_node=2 $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
 