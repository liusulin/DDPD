#!/bin/bash
GPU_ID=0
GPT_DIR='/pscratch/sd/s/sulinl/gptj'
DATA_DIR='/pscratch/sd/s/sulinl/dfm/data/text8'
OUT_DIR='/pscratch/sd/s/sulinl/ddpd_eval/text8'
OUT_FOLDER='dpdd-double-planner-uni-denoiser'

# Function to evaluate samples
evaluate_samples() {
    local out_dir="$OUT_DIR/$OUT_FOLDER"
    CUDA_VISIBLE_DEVICES=$GPU_ID python text8/eval/sample_eval.py --path=$out_dir --cache_dir=$GPT_DIR --data_dir=$DATA_DIR
}

evaluate_samples