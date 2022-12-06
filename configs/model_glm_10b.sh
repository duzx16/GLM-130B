MODEL_TYPE="glm-10b"
CHECKPOINT_PATH="/zhangpai21/checkpoints/blocklm-10b-mip"
MP_SIZE=1
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --vocab 50304 \
            --num-layers 48 \
            --hidden-size 4096 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --task-mask \
            --use-gpu-initialization \
            --load ${CHECKPOINT_PATH}"
