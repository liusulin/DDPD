"""This file contains implementation for DDPD sampling.

Reference: 
    https://github.com/huggingface/open-muse
    https://github.com/baaivision/MUSE-Pytorch
    https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/maskgit.py
"""

import torch
from torch import nn
import numpy as np
import math
import torch.utils.checkpoint
from transformers import BertConfig, BertModel

import json
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from pathlib import Path

from .blocks import UViTBlock


class DDPD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, denoiser, planner, codebook_size, image_seq_len):
        super().__init__()
        self.codebook_size = codebook_size
        self.image_seq_len = image_seq_len
        self.denoiser = denoiser
        self.planner = planner

    @torch.no_grad()
    def generate(self,
                 condition,
                 scheduler="arccos",
                 guidance_scale=3.0,
                 planner_guidance_scale=0.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 softmax_temperature_annealing=False,
                 conf_anneal=True,
                 logit_temp_anneal=True,
                 randomize_temperature=4.5,
                 logit_temperature=1.0,
                 conf_method="logit",
                 num_sample_steps=8,
                 num_refinement_steps=0,
                 refine_delta=5,
                 conf_tol_min=-100,
                 debug=False):
        if guidance_decay not in ["constant", "linear", "power-cosine"]:
            # contstant: constant guidance scale
            # linear: linear increasing the guidance scale as in MUSE
            # power-cosine: the guidance schedule from MDT
            raise ValueError(f"Unsupported guidance decay {guidance_decay}")
        device = condition.device
        # start with random noise tokens
        ids = torch.randint(0, self.codebook_size, (condition.shape[0], self.image_seq_len), device=device)
        full_mask_values = torch.full((condition.shape[0], self.image_seq_len), self.codebook_size, device=device)
        
        cfg_scale = guidance_scale if guidance_decay == "constant" else 0.
        cfg_planner_scale = planner_guidance_scale if guidance_decay == "constant" else 0.

        # Add gumbel noise
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))
        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))
        def add_gumbel_noise(t, temperature):
            return t + temperature * gumbel_noise(t)
                
        mask_len_previous = self.image_seq_len
        noise_logits = torch.full(ids.shape, np.inf, device=device) # for the first step, all tokens are noise, set noise logits to inf
        
        for step in range(num_sample_steps + num_refinement_steps):
            noise_prob = torch.sigmoid(noise_logits)
            noise_prob_sum = noise_prob.sum(-1)            
            mask_pred = torch.bernoulli(noise_prob).bool() # mask prediction, when denoising, masks should be applied

            ratio = 1 - noise_prob_sum.mean() / self.image_seq_len # use the actual ratio based on data for annealing
            annealed_temp = (randomize_temperature * (1.0 - ratio)) if conf_anneal else randomize_temperature
            logit_annealed_temp = (logit_temperature * (1.0 - ratio)).clamp(min=1.0) if logit_temp_anneal else logit_temperature

            if guidance_decay == "power-cosine":
                # ref: https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py#L501
                guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((ratio) ** guidance_scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step
                cfg_planner_scale = (planner_guidance_scale - 1) * scale_step
            elif guidance_decay == "linear":
                cfg_scale = ratio * guidance_scale
                cfg_planner_scale = ratio * planner_guidance_scale

            if step < num_sample_steps:
                ratio_mask = 1. * (step + 1) / (num_sample_steps) # use prescheduled mask ratio
                if scheduler == "arccos":
                    mask_ratio = np.arccos(ratio_mask) / (math.pi * 0.5)
                elif scheduler == "linear":
                    mask_ratio = 1 - ratio_mask
                else:
                    raise ValueError(f"Unsupported scheduler {scheduler}")
                mask_len = torch.Tensor([np.floor(self.image_seq_len * mask_ratio)]).to(device)
                unmask_increment = mask_len_previous - mask_len
                unmask_increment = (unmask_increment * noise_prob_sum / mask_len_previous)
                unmask_increment = unmask_increment.clamp(min=1).long()
                mask_len_previous = mask_len
            else:
                unmask_increment = noise_prob_sum.long() # if in refinement stage, use the actual number of all noisy tokens to be denoised
                unmask_increment = unmask_increment.clamp(min=refine_delta) # set a fixed number of tokens to be denoised if actual predicted noisy tokens are less than refine_delta
            
            # figure out which tokens to be denoised
            confidence = add_gumbel_noise(noise_prob.log(), 1.0) # set to 1.0 for gumbel softmax sampling means select tokens proportionally to the noise probability
            sorted_confidence, _ = torch.sort(confidence, axis=-1, descending=True)
            cut_off = sorted_confidence[torch.arange(sorted_confidence.shape[0]), unmask_increment-1]
            if_denoise = torch.logical_and(confidence >= cut_off.view(-1, 1), noise_logits > conf_tol_min)
            is_masked = torch.logical_or(mask_pred, if_denoise)
            ids_masked = torch.where(is_masked, full_mask_values, ids)
            
                
            if cfg_scale > 0:
                cond_logits = self.denoiser(
                    ids_masked, condition, cond_drop_prob=0.0
                )
                uncond_logits = self.denoiser(
                    ids_masked, condition, cond_drop_prob=1.0
                )
                logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.denoiser(
                    ids_masked, condition, cond_drop_prob=0.0
                )
            if softmax_temperature_annealing:
                softmax_temperature = 0.5 + 0.8 * (1 - ratio)
                logits = logits / softmax_temperature
                
            sampled_ids = add_gumbel_noise(logits, logit_annealed_temp).argmax(dim=-1)                    
            sampled_ids = torch.where(is_masked, sampled_ids, ids)
            if conf_method == "logit": # use logit confidence based selection of which tokens to be denoised, as in MaskGit
                sampled_logits = torch.squeeze(
                    torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
                confidence = add_gumbel_noise(sampled_logits, annealed_temp) # do not use logit_noise here, because there might be some good tokens being masked again
                confidence = torch.where(is_masked, confidence, -np.inf)
            elif conf_method == "random":
                pass
            else:
                raise ValueError(f"Unsupported confidence type {conf_method}")        
                
            sorted_confidence, _ = torch.sort(confidence, axis=-1, descending=True)
            cut_off = sorted_confidence[torch.arange(sorted_confidence.shape[0]), unmask_increment-1]
            if_denoise = torch.logical_and(confidence >= cut_off.view(-1, 1), noise_logits > conf_tol_min)
            if step == num_sample_steps + num_refinement_steps - 1:
                if_denoise = is_masked.clone()  # for the last step, all tokens are denoised
            ids = torch.where(if_denoise, sampled_ids, ids)

        return ids