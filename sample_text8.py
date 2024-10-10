import os
import time
import math
import pickle
from contextlib import nullcontext
from typing import Optional
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import uuid

import time
# -----------------------------------------------------------------------------
# These configs will be overridden by the config file and so their values here do not matter.
out_dir = "out"
out_folder = "out-text8-eval"
run_name = "sample"
# data
dataset = "text8"
batch_size = 64
block_size = 256

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
qk_layernorm = True

# for ce loss on x0 evaluation
eval_iters = 200


# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.

# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
data_dir = "data/text8"  #  directory should contain meta.pkl

# sampling
total_samples = 128
x1_temp = 1.0

planner_ckpt_path = "out/ckpt.pt"
denoiser_ckpt_path= 'empty'
is_mask_denoiser = False
# dpd settings
timesteps = 1000
final_fill_in_prob = 0.05
use_softmax_for_dim_denoise = True
fixed_time = False
chosen_time = 0.85


# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("text8/configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


out_dir = os.path.join(out_dir, out_folder)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
hash = str(uuid.uuid1()).split("-")[0]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
samples_dir = os.path.join(
    out_dir, "samples_" + f"temp_{x1_temp:.1f}_"+ time.strftime("%Y-%m-%d-%H-%M-%S") + "_" + hash
)
os.mkdir(samples_dir)
with open(os.path.join(samples_dir, "config.yaml"), "w") as f:
    yaml.dump(config, f, sort_keys=False)

with open(os.path.join(samples_dir, f"run_name_{run_name}.txt"), "w") as f:
    f.write(f"{run_name}")



meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

stoi = meta["stoi"]
itos = meta["itos"]

if dataset == "text8":
    # increase vocab size by 1 to include a mask token
    meta_vocab_size += 1
    mask_token_id = meta_vocab_size - 1
    stoi["X"] = mask_token_id
    itos[mask_token_id] = "X"
else:
    raise NotImplementedError


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


### LOAD MODEL
device_type = "cuda"
device = "cuda:0"

def load_model(ckpt_path, is_denoiser=False):
    # resume training from a checkpoint.
    print(f"Loading network from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    model_args["vocab_size"] = meta_vocab_size
    if is_denoiser:
        from text8.model_denoiser import GPT, GPTConfig
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        from text8.model_planner import GPT, GPTConfig
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    checkpoint["model_args"] = model_args
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return model, checkpoint


planner_model, planner_checkpoint = load_model(planner_ckpt_path)

# save the model information to the sample directory
model_information = {
    "model_args": planner_checkpoint["model_args"],
}
torch.save(model_information, os.path.join(samples_dir, "planner_model_information.pt"))
planner_checkpoint = None
planner_model.eval()
planner_model.to(device)

denoiser_model, denoiser_checkpoint = load_model(denoiser_ckpt_path, is_denoiser=True)
if is_mask_denoiser:
    assert denoiser_model.config.model_type == 'ddpd_denoiser_mask'
else:
    assert denoiser_model.config.model_type == 'ddpd_denoiser_uniform'
print("Denoiser type: ", denoiser_model.config.model_type)
model_information = {
    "model_args": denoiser_checkpoint["model_args"],
}
torch.save(model_information, os.path.join(samples_dir, "denoiser_model_information.pt"))
denoiser_checkpoint = None
denoiser_model.eval()
denoiser_model.to(device)

if compile:
    print("compiling the planner model... (takes a ~minute)")
    planner_model = torch.compile(planner_model)  # requires PyTorch 2.0
    print("compiling the denoiser model... (takes a ~minute)")
    denoiser_model = torch.compile(denoiser_model)
        

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
print(torch.__version__)
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# ----------------- DATA LOADING CODE --------------=-
# poor man's data loader
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split, times=None):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # sample random batch
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if times is None:
        times = torch.randint(low=1, high=timesteps+1, size=(batch_size,))
    else:
        assert times.shape == (batch_size,)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, times = x.pin_memory().to(device, non_blocking=True), \
            y.pin_memory().to(device, non_blocking=True), \
            times.pin_memory().to(device, non_blocking=True)
    else:
        x, y, times = x.to(device), y.to(device), times.to(device)
    return x, y, times

# ----------------- SAMPLING CODE --------------=-

S = meta_vocab_size - 1
B = batch_size
D = block_size

# write an empty file to store the samples eventually
with open(os.path.join(samples_dir, "samples.txt"), "w") as f:
    pass

assert total_samples % B == 0


with torch.no_grad():
    with ctx:
        for _ in range(total_samples // B):
            D = denoiser_model.config.block_size
            S = denoiser_model.config.mask_token_id
            samples = torch.randint(0, S, (B, D), dtype=torch.int64, device=device)
            ones = torch.ones((B, ), dtype=torch.int64, device=device)
            time_profile = time.time()
            dummy_times = 0.0*ones
            for t in np.arange(timesteps, 0, -1):
                with torch.no_grad():
                    logits_if_noise = planner_model._run_net(samples, dummy_times)
                # get probability of being noise
                prob_noise = F.sigmoid(logits_if_noise.squeeze()) # (B, D)
                noise_count = prob_noise.sum(-1) # (B,)
                # always make a transition otherwise we waste a NFE
                if use_softmax_for_dim_denoise:
                    prob_change_d = F.softmax(logits_if_noise.squeeze()/x1_temp, dim=-1)
                    dim_change = torch.multinomial(prob_change_d, 1).squeeze() # (B,)
                else:
                    dim_change = torch.multinomial(prob_noise, 1).squeeze() # (B,)
                
                with torch.no_grad():
                    if denoiser_model.config.model_type == 'ddpd_denoiser_mask': # logit from a model that take in masked data
                        # sample whether to mask based on prob_noise
                        mask = torch.bernoulli(prob_noise).bool().long() # (B, D)
                        mask[torch.arange(B), dim_change] = 1 # always mask the point that is changing
                        # apply mask to the data
                        samples_input = samples * (1 - mask) + mask_token_id * mask
                        times_input = 1 - mask.sum(-1)/D # (B,)
                        logits_x1 = denoiser_model._run_net(samples_input.long(), times_input)
                    else: # logit from a model that take in noisy data
                        times_input = 1 - noise_count/D # (B,)
                        logits_x1 = denoiser_model._run_net(samples, times_input)
                    probs_x1 = F.softmax(logits_x1/x1_temp, dim=-1)

                            
                probs_x1[:, :, mask_token_id] = 0.0 # do not allow to sample mask token
                probs_change = probs_x1[torch.arange(B), dim_change, :] # (B, S)
                x1_values = torch.multinomial(probs_change, 1).squeeze() # (B,)
                samples[torch.arange(B), dim_change] = x1_values
                
                if (timesteps-t) % 100 == 0:
                    samples_np = samples.cpu().detach().numpy()
                    for sample_idx in range(5):
                        with open(os.path.join(samples_dir, 'samples_traj.txt'), 'a') as f:
                            f.write('timestep ' + str(t) + '\n')
                            f.write(decode(samples_np[sample_idx]) + '\n')
                    print(f"sampled {timesteps-t} timesteps")
                    print(f"average noise count: {noise_count.mean()}")
                    with open(os.path.join(samples_dir, f"run_name_{run_name}.txt"), "a") as f:
                        f.write(f"sampling {timesteps-t} timesteps\n")
                        f.write(f"average noise count: {noise_count.mean()}\n")
            
            fill_in_positions = prob_noise > final_fill_in_prob # (B, D)
            x1_values = torch.argmax(logits_x1, dim=-1) # (B, D)
            samples[fill_in_positions] = x1_values[fill_in_positions]
            
            logits_if_noise = planner_model._run_net(samples, times_input)
            prob_noise = F.sigmoid(logits_if_noise.squeeze()) # (B, D)
            noise_count = prob_noise.sum(-1) # (B,)
            
            samples_np = samples.cpu().detach().numpy()
            for sample_idx in range(samples_np.shape[0]):
                with open(os.path.join(samples_dir, "samples.txt"), "a") as f:
                    f.write(decode(samples_np[sample_idx]) + "\n")
            with open(os.path.join(samples_dir, f"run_name_{run_name}.txt"), "a") as f:
                f.write(f"sampling {timesteps-t} timesteps\n")
                f.write(f"average noise count: {noise_count.mean()}\n")
            print(f"sampled {timesteps-t} timesteps")
            print(f"average noise count: {noise_count.mean()}")
          

with open(os.path.join(samples_dir, "finished_sampling.txt"), "w") as f:
    f.write("finished sampling\n")
