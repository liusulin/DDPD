"""Sampling scripts for TiTok on ImageNet.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: 
    https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""

import demo_util
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import os
import math
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_image(sample_dir, i):
    sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
    sample_np = np.asarray(sample_pil).astype(np.uint8)
    return sample_np

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples using parallelization.
    """
    samples = []    
    # Use ThreadPoolExecutor for parallel image loading and processing
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, sample_dir, i): i for i in range(num)}
        for future in tqdm(as_completed(futures), total=num, desc="Building .npz file from samples"):
            sample_np = future.result()
            samples.append(sample_np)
            i = futures[future]
            if i > 100:
                # delete all samples except the first 100
                os.remove(f"{sample_dir}/{i:06d}.png")
    
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main():
    config = demo_util.get_config_cli()
    num_fid_samples = 50000
    per_proc_batch_size = 250
    hyp = config.model.generator
    sample_folder_dir = (
        f"{config.experiment.output_dir}/{hyp.num_steps}steps_"
        f"cfg_{hyp.guidance_scale}_conf_{hyp.conf_method}_logitT_{hyp.logit_temperature}_randT_{hyp.randomize_temperature}_"
        f"{hyp.guidance_decay}_r{hyp.num_refine_steps}_d{hyp.refine_delta}_tol_{hyp.conf_tol_min}_"
        f"softmaxAnneal{hyp.softmax_temperature_anneal}_logitAnneal{hyp.logit_temp_anneal}_confAnneal{hyp.conf_anneal}"
    )
    npz_path = f"{sample_folder_dir}.npz"
    # skip if npz_path file already exists
    if os.path.exists(npz_path):
        print(f"Skipping sampling as {npz_path} already exists.")
        return
    
    seed = 42
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # setup DDP.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size() 
    device = rank % torch.cuda.device_count()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"Sampling {num_fid_samples} images with hyperpameters: {hyp}")
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.") 

    # if rank == 0:
        # # downloads from hf
        # hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.experiment.tokenizer_checkpoint}", local_dir="./")
        # hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.experiment.generator_checkpoint}", local_dir="./")
    dist.barrier()

    titok_tokenizer = demo_util.get_titok_tokenizer(config)
    titok_generator = demo_util.get_titok_generator(config)
    planner = demo_util.get_titok_planner(config)
    titok_tokenizer.to(device)
    titok_generator.to(device)
    planner.to(device)
    
    titok_generator = torch.compile(titok_generator)
    titok_tokenizer = torch.compile(titok_tokenizer)
    planner = torch.compile(planner)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    assert num_fid_samples % global_batch_size == 0
    if rank == 0:
        print(f"Total number of images that will be sampled: {num_fid_samples}")

    samples_needed_this_gpu = int(num_fid_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    all_classes = list(range(config.model.generator.condition_num_classes)) * (num_fid_samples // config.model.generator.condition_num_classes)
    subset_len = len(all_classes) // world_size
    all_classes = np.array(all_classes[rank * subset_len: (rank+1)*subset_len], dtype=np.int64)
    cur_idx = 0

    for _ in pbar:
        y = torch.from_numpy(all_classes[cur_idx * n: (cur_idx+1)*n]).to(device)
        cur_idx += 1

        samples = demo_util.ddpd_sample_fn(
            generator=titok_generator,
            planner=planner,
            tokenizer=titok_tokenizer,
            labels=y.long(),
            guidance_scale=config.model.generator.guidance_scale,
            guidance_decay=config.model.generator.guidance_decay,
            randomize_temperature=config.model.generator.randomize_temperature,
            conf_method=config.model.generator.conf_method,
            logit_temperature=config.model.generator.logit_temperature,
            conf_anneal=config.model.generator.conf_anneal,
            logit_temp_anneal=config.model.generator.logit_temp_anneal,
            softmax_temperature_annealing=config.model.generator.softmax_temperature_anneal,
            num_sample_steps=config.model.generator.num_steps,
            num_refinement_steps=config.model.generator.num_refine_steps,
            refine_delta=config.model.generator.refine_delta,
            conf_tol_min=config.model.generator.conf_tol_min,
            device=device
        )
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()