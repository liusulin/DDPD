"""
Extract features from ImageNet dataset using the titok tokenizer.
Reference: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import os
import json

import demo_util


#################################################################################
#                             Helper Functions                         #
#################################################################################


def cleanup():
    """
    End DDP.
    """
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Extract tokens Loop                                #
#################################################################################

def main(args):
    """
    Extract image tokens from ImageNet dataset using the titok tokenizer.
    """
    assert torch.cuda.is_available(), "requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    config = demo_util.get_config(args.config_path)
    titok_tokenizer = demo_util.get_titok_tokenizer(config).to(device)
    latent_size = (args.config_path).split('/')[-1].split('_')[1].split('.')[0]
    
    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'imagenet{args.image_size}_features_{latent_size}'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'imagenet{args.image_size}_labels_{latent_size}'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'imagenet{args.image_size}_reconstructed_images_{latent_size}'), exist_ok=True)
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    if rank == 0:
        # save class to index mapping to json file
        class_to_idx = dataset.class_to_idx
        with open(f'{args.features_path}/class_to_idx.json', 'w') as f:
            json.dump(class_to_idx, f)
            
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x, y in loader:
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            x = titok_tokenizer.encode(x)[1]["min_encoding_indices"] # shape (1, 1, 128)        
        
        # optional: save reconstructed image to check the quality of the tokenizer
        
        # with torch.no_grad():
        #     reconstructed_image = titok_tokenizer.decode_tokens(x)
        # reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
        # reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        # reconstructed_image = Image.fromarray(reconstructed_image)
        # folder_num = train_steps // 10000
        # if train_steps % 10000 == 0:
        #     os.makedirs(f'{args.features_path}/imagenet{args.image_size}_reconstructed_images_{latent_size}/{folder_num}', exist_ok=True)
        # reconstructed_image.save(f'{args.features_path}/imagenet{args.image_size}_reconstructed_images_{latent_size}/{folder_num}/{train_steps}.png')
        
        x = x.detach().cpu().numpy()    
        np.save(f'{args.features_path}/imagenet{args.image_size}_features_{latent_size}/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.features_path}/imagenet{args.image_size}_labels_{latent_size}/{train_steps}.npy', y)
            
        train_steps += 1
        print(train_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/titok_s128.yaml")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
