#!/bin/bash

# Set script to exit on any errors.
set -e
GPT_DIR='/pscratch/sd/s/sulinl/gptj'
DATA_DIR='/pscratch/sd/s/sulinl/dfm/data/text8'
OUT_DIR='/pscratch/sd/s/sulinl/ddpd_eval/text8'
OUT_FOLDER='dpdd-double-planner-uni-denoiser'
# Define constants and configurations
PLANNER_CKPT='/pscratch/sd/s/sulinl/ddpd_pretrained/text8/planner_model.pt'
# '/pscratch/sd/s/sulinl/dpd/out-text8/2024-04-27-6w2cgc92_dpd_v3_no_te_pred_mask/ckpt_450000.pt'
DENOISER_CKPT='/pscratch/sd/s/sulinl/ddpd_pretrained/text8/denoiser_model_uniformD.pt'
# '/pscratch/sd/s/sulinl/dfm/out-text8/2024-04-23-gklqyh74_dfm_uniform_nosc/best_ckpt.pt'
TEMPERATURES=(1.0 0.9 0.8)
RUN_NAMES=("temp_1" "temp_09" "temp_08")

IS_MASK_DENOISER=False
# Array of GPUs to use
GPU_ID=1
STEPS=1000
# Number of sampling experiments to run in parallel
PARALLEL=3

# Cleanup function
cleanup() {
    echo "Interrupt received, stopping all processes..."
    # Kill all child processes of this script
    pkill -P $$
    exit 1
}

trap 'cleanup' INT


# Function to run the sampling command with a run name
run_sampling() {
    local temp=$1
    local run_name=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID python sample_text8.py text8/config/sample.py --x1_temp=$temp --run_name=$run_name\
     --out_folder=$OUT_FOLDER --out_dir=$OUT_DIR --planner_ckpt_path=$PLANNER_CKPT --denoiser_ckpt_path=$DENOISER_CKPT \
     --is_mask_denoiser=$IS_MASK_DENOISER --timesteps=$STEPS &
}


# Subshell trap setup for handling interrupts
trap 'echo "Interrupted, stopping background tasks..."; kill 0; exit' INT


for i in "${!TEMPERATURES[@]}"; do
    run_sampling ${TEMPERATURES[$i]} ${RUN_NAMES[$i]}
    let COUNTER+=1
    if (( COUNTER % PARALLEL == 0 )); then
        wait
    fi
done
if (( COUNTER % PARALLEL != 0 )); then
    wait
fi