#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_10b.sh"

DATA_PATH="/zhangpai21/workspace/zxdu/data"

ARGS="${main_dir}/evaluate.py \
       --mode inference \
       --data-path $DATA_PATH \
       --task $* \
       $MODEL_ARGS"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${TIMESTAMP}

mkdir -p logs

run_cmd="torchrun --nproc_per_node 8 ${ARGS}"
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}.log
