import torch
import torch.multiprocessing as mp
import argparse
import time

from load_model import load_model, load_model_local_planner
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
import utils

import numpy as np
import os
import glob

def generate_samples(gpu_id, start_index, end_index, args):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    denoiser_model, mask_graph, noise = load_model(args.denoiser_model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    if args.method == 'sedd':
        sampling_fn = sampling.get_pc_sampler(
            mask_graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
        )
        samples = sampling_fn(denoiser_model)
    elif args.method == 'ddpd':
        planner_model, uniform_graph, uniform_noise = load_model_local_planner(args.planner_model_path, device)
        sampling_fn = sampling.get_ddpd_sampler(
            mask_graph, uniform_graph, noise, (args.batch_size, 1024), 'analytic', args.steps, 
            top_p=args.top_p, device=device, use_prob_for_dim_change=args.use_prob_for_dim_change,
        )
        samples = sampling_fn(denoiser_model, planner_model)
    else:
        raise ValueError(f"Unknown method {args.method}")

    text_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    
    index = range(start_index, end_index)
    for i, text in enumerate(text_samples):
        name_file = f"samples_{index[i]}.txt"
        file_path = os.path.join(args.folder_path, name_file)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--method", default='ddpd', type=str)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--denoiser_model_path", "--model_path", default="louaaron/sedd-small", type=str)
    parser.add_argument("--gen_sample_path", default="path/to/gen/sample", type=str)
    parser.add_argument("--planner_model_path", default="path/to/planner/model", type=str)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--use_prob_for_dim_change", action='store_true')
    parser.add_argument("--num_processes", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()


    denoiser_model_name = args.denoiser_model_path.split('/')[1]
    method_folder_name = f"gen-denoiser-{denoiser_model_name}"
    if args.use_prob_for_dim_change:
        method_folder_name = f"{method_folder_name}-prob"
    folder_name = f"{args.method}_{args.steps}_p_{args.top_p}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    args.folder_path = os.path.join(args.gen_sample_path, method_folder_name, folder_name)
    utils.makedirs(args.folder_path)
    print("saving generated samples to ", args.folder_path)
    with open(os.path.join(args.folder_path, "args.text"), 'w') as f:
        f.write(str(args))
    
    processes = []
    for gpu_id in range(args.num_processes):
        start_index = gpu_id * args.batch_size
        end_index = start_index + args.batch_size
        p = mp.Process(target=generate_samples, args=(gpu_id, start_index, end_index, args))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
