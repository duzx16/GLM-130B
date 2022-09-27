#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

DATESTR=$(date +"%m-%d-%H-%M")
source "${main_dir}/configs/model_glm_130b_int8.sh"

#DATA_PATH="/zhangpai21/workspace/zxdu"
DATA_PATH="/group/30042/zhengxiaodu"

ARGS="${main_dir}/evaluate.py \
       --mode inference \
       --data-path $DATA_PATH \
       --task $* \
       $MODEL_ARGS"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${TIMESTAMP}

mkdir -p logs

mkdir logs
run_cmd="torchrun --nproc_per_node $MP_SIZE ${ARGS} 2>&1 | tee logs/log-${DATESTR}.txt"
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}.log
