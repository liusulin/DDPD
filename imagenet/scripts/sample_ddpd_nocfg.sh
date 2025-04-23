#!/bin/bash

# Define single preset values instead of arrays
guidance_scale=0.0 # no CFG 
output_dir="path/to/output/folder"
num_steps=16
num_refine_steps=8
randomize_temperature=1.0
logit_temperature=1.0
guidance_decay="linear"
refine_delta=5
logit_temp_anneal=False
conf_anneal=False
softmax_temperature_anneal=False
conf_method="random"
conf_tol_min=-6.0 # planner logit threshold for if needs denoising
python_script="sample_imagenet_ddpd.py"

# Get the node list
NODE_LIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -d, -s)
echo "Node list: $NODE_LIST"

# Determine the master node (first node in the list)
MASTER_NODE=$(echo $NODE_LIST | cut -d, -f1)
echo "Master node: $MASTER_NODE"

# Run the sampling command
torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE \
--rdzv_backend=c10d --rdzv_endpoint=$MASTER_NODE:29507 $python_script \
config=configs/s128_ddpd_sample.yaml model.generator.num_steps=${num_steps} \
experiment.output_dir=${output_dir} model.generator.guidance_decay=${guidance_decay} \
model.generator.num_refine_steps=${num_refine_steps} model.generator.refine_delta=${refine_delta} \
model.generator.logit_temperature=${logit_temperature} \
model.generator.logit_temp_anneal=${logit_temp_anneal} \
model.generator.conf_anneal=${conf_anneal} \
model.generator.softmax_temperature_anneal=${softmax_temperature_anneal} \
model.generator.randomize_temperature=$randomize_temperature \
model.generator.guidance_scale=$guidance_scale \
model.generator.conf_method=$conf_method \
model.generator.conf_tol_min=$conf_tol_min