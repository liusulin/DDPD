"""
Training script for training the planner in DDPD.
Reference: https://github.com/facebookresearch/DiT/blob/main/train.py
           https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
from PIL import Image
from glob import glob
from time import time
import logging
import os
import time
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from pathlib import Path
import wandb
from typing import Any, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
from accelerate.utils import DataLoaderConfiguration
from accelerate.utils.other import is_compiled_module

from modeling.maskgit import UViTPlanner, ViTPlanner
from modeling.ema import ExponentialMovingAverage
import demo_util
from training_util import get_lr, log_image_and_pred, AverageMeter

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(is_main_process):
    """
    Create a logger that writes to stdout.
    """
    logger = logging.getLogger(__name__)
    
    if is_main_process:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler()]
        )
    else:  # log only errors
        logging.basicConfig(
            level=logging.ERROR,
            format='[\033[31m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler()]
        )
    
    return logger


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


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def mask_image(input_ids, codebook_size, fixed_time=False, schedule='arccos', max=1.0, min=0.0):
    batch_size, length = input_ids.shape
    if fixed_time:
        time_point = min + 0.1
        mask = torch.rand((batch_size, length), device=input_ids.device) < time_point
        t = torch.full((batch_size, 1), time_point, device=input_ids.device)
    else:
        t = torch.rand((batch_size, 1), device=input_ids.device)
        if schedule == 'arccos':
            t = torch.arccos(t) / (np.pi * 0.5) * (max - min) + min
        elif schedule == 'linear':
            t = t * (max - min) + min
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        mask = torch.rand((batch_size, length), device=input_ids.device) < t
    noise_ids = torch.randint(0, codebook_size, (batch_size, length), device=input_ids.device)
    input_ids = torch.where(mask, noise_ids, input_ids)
    return input_ids, mask, t
    
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main():
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    config = demo_util.get_config_cli()

    dataloaderconfig = DataLoaderConfiguration(non_blocking=True)
    # Setup accelerator:
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.results_dir,
        dataloader_config=dataloaderconfig,
    )
    config.training.batch_size = int(config.training.global_batch_size // accelerator.num_processes)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.batch_size
        )

    logger = create_logger(accelerator.is_main_process)
    
    if config.training.seed is not None:
        set_seed(config.training.seed)
    # Create model:
    codebook_size = config.model.vq_model.codebook_size
    mask_schedule = config.training.mask_schedule
    mask_rate_max = config.training.mask_rate_max
    mask_rate_min = config.training.mask_rate_min
    assert config.dataset.preprocessing.crop_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if config.model.planner.model_type == 'UViT':
        model = UViTPlanner(config)
    elif config.model.planner.model_type == 'ViT':
        model = ViTPlanner(config)
    
    tokenizer = demo_util.get_titok_tokenizer(config).to(accelerator.device)
    
    if accelerator.is_main_process:
        logger.info(f"UViT/ViT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    total_batch_size = (
        config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    # Setup data:
    features_dir = f"{config.training.feature_path}_features"
    labels_dir = f"{config.training.feature_path}_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({config.training.feature_path})")
        
    model = model.to(accelerator.device)
    ema = ExponentialMovingAverage(model.parameters(), config.training.ema_decay)

    if config.experiment.resume_from_checkpoint:
        ckpt_path = config.experiment.resume_checkpoint_path
        if ckpt_path is None:
            raise ValueError("Please specify a checkpoint path to resume training from.")
        else:
            accelerator.print(f"Resuming training from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=accelerator.device)
        state_dict = checkpoint["model"]
        # fix the state_dict keys if they have the unwanted prefix
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        train_steps = checkpoint["iter"]
    else:
        train_steps = 0
        if config.experiment.warmup_from_generator:
            state_dict = torch.load(config.experiment.generator_checkpoint, map_location=accelerator.device)
            # remove params that start with lm_head
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head")}
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.lm_head")}
            model.load_state_dict(state_dict, strict=False)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        resume_run = config.experiment.resume_from_checkpoint
        if not resume_run:
            run_id = wandb.util.generate_id()
            config.experiment.wandb_run_id = run_id
            generator_config = config.model.generator
            experiment_index = len(glob(f"{config.experiment.results_dir}/*"))
            model_string_name = f"{config.model.planner.model_type}-{generator_config.image_seq_len}-{generator_config.num_hidden_layers}L-{generator_config.num_attention_heads}H-{generator_config.hidden_size}D"
            config.experiment.out_dir = f"{config.experiment.results_dir}/{experiment_index:03d}-{model_string_name}-{run_id}"  # Create an experiment folder 
            config_path = Path(config.experiment.out_dir) / "config.yaml"
        else:
            run_id = config.experiment.wandb_run_id
            assert run_id is not None, "When resuming, must provide the run_id of the run to resume from."
            config.experiment.out_dir = os.path.dirname(config.experiment.resume_checkpoint_path)
            config_path = Path(config.experiment.out_dir) / "config_resume.yaml"
        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_run,
            config_exclude_keys=[],
            dir=f"{config.experiment.results_dir}",
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        

        os.makedirs(config.experiment.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(config.experiment.out_dir, exist_ok=True)
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # Prepare models for training:
    model, opt, loader = accelerator.prepare(model, opt, loader)

    if config.training.compile:
        model = torch.compile(model)
    logger.info(f"Model prepared is compiled: {is_compiled_module(model)}")        
    raw_model = model if accelerator.num_processes == 1 else model.module

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Start training from here:
    if accelerator.is_main_process:
        logger.info("************** Running training ***************")
        logger.info(f"  Num training steps = {config.training.iter}, starting from step {train_steps}")
        logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
        logger.info(f"  Instantaneous batch size per device = {config.training.batch_size}")
        logger.info(f"  Global train batch size (w. parallel, distributed) = {config.training.global_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    while train_steps < config.training.iter:
        for ids, y in loader:
            # set learning rate for this iteration
            lr = get_lr(train_steps+1, config.training)
            for param_group in opt.param_groups:
                param_group["lr"] = lr
                
            with accelerator.accumulate(model):
                ids = ids.squeeze(dim=1).squeeze(dim=1) # (B, L)
                ids_masked, mask, mask_prob = mask_image(
                    ids, codebook_size, fixed_time=False, schedule=mask_schedule,
                    max=mask_rate_max, min=mask_rate_min)
                data_time_m.update(time.time() - end)
                
                logit_noise = model(ids_masked, y, cond_drop_prob=config.training.cond_drop_prob)
                
                if accelerator.is_main_process and train_steps == 0:
                    logger.info(f"Beginning from step {train_steps}...")
                    logger.info(f"Learning rate: {opt.param_groups[0]['lr']:.7e}")
                    logger.info(f"Input shape: {ids.shape}")
                    logger.info(f"Target shape: {y.shape}") 
                    logger.info(f"Mask shape: {mask.shape}")
                    logger.info(f"logit_noise shape: {logit_noise.shape}")
                

                loss = binary_cross_entropy_with_logits(logit_noise, mask.float())
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                opt.step()
                opt.zero_grad()
                ema.update(raw_model.parameters())

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (train_steps + 1) % config.experiment.log_every == 0 and accelerator.is_main_process:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                    )
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr,
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=train_steps + 1)

                    logger.info(
                        f"Step: {train_steps + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr:.6e}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                    
                
                if (train_steps + 1) % config.experiment.eval_every == 0 and accelerator.is_main_process:
                    ema.store(raw_model.parameters())
                    ema.copy_to(raw_model.parameters())
                    
                    model.eval()
                    # optional: log the mask and target
                    # ids_masked, mask, mask_prob = mask_image(
                    #     ids, codebook_size, fixed_time=False, schedule=mask_schedule,
                    #     max=mask_rate_max, min=mask_rate_min)
                    # with torch.no_grad():
                    #     logit_noise = model(ids_masked, y, cond_drop_prob=0.0)

                    # pred_noise = torch.sigmoid(logit_noise)
                    # log_image_and_pred(tokenizer, ids, ids_masked, mask, pred_noise, train_steps+1, n_sample=10, fixed_time=False, num_tokens=mask.shape[-1])
                    # ids_masked, mask, mask_prob = mask_image(
                    #     ids, codebook_size, fixed_time=True, schedule=mask_schedule,
                    #     max=mask_rate_max, min=mask_rate_min)
                    # with torch.no_grad():
                    #     logit_noise = model(ids_masked, y, cond_drop_prob=0.0)
                    # pred_noise = torch.sigmoid(logit_noise)
                    # log_image_and_pred(tokenizer, ids, ids_masked, mask, pred_noise, train_steps+1, n_sample=10, fixed_time=True, num_tokens=mask.shape[-1])
                    
                    ema.restore(raw_model.parameters())
                    model.train()
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "config": config,
                            "iter": train_steps + 1,
                        }
                        checkpoint_path = f"{config.experiment.out_dir}/current.pt"
                        torch.save(checkpoint, checkpoint_path)
                        
                # Save checkpoint:
                if (train_steps + 1) % config.experiment.ckpt_every == 0 and accelerator.is_main_process:
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "config": config,
                            "iter": train_steps + 1,
                        }
                        checkpoint_path = f"{config.experiment.out_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                train_steps += 1

    # model.eval()  # important! This disables randomized embedding dropout
    # # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = {
            "model": raw_model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "config": config,
            "iter": train_steps,
        }
        checkpoint_path = f"{config.experiment.out_dir}/final.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved final checkpoint to {checkpoint_path}")
        logger.info("Done!")
    accelerator.end_training()


if __name__ == "__main__":
    main()
