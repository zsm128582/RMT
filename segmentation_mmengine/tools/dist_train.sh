CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
BASE_PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 给端口加一个随机偏移，范围 0-99
RAND_OFFSET=$((RANDOM % 100))
PORT=$((BASE_PORT + RAND_OFFSET))

echo "Using random master port: $PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py --resume \
    $CONFIG \
    --launcher pytorch ${@:3}
