"""Demo file for sampling images from TiTok.

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
"""


import torch

from omegaconf import OmegaConf
from modeling.titok import TiTok
from modeling.maskgit import ImageBert, UViTBert, UViTPlanner, ViTPlanner
from modeling.maskgit_ddpd import DDPD

def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf

def get_titok_tokenizer(config):
    tokenizer = TiTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_titok_generator(config):
    if config.model.generator.model_type == "ViT":
        model_cls = ImageBert
    elif config.model.generator.model_type == "UViT":
        model_cls = UViTBert
    else:
        raise ValueError(f"Unsupported model type {config.model.generator.model_type}")
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

def get_titok_planner(config):
    if config.model.planner.model_type == "UViT":
        model_cls = UViTPlanner
    elif config.model.planner.model_type == "ViT":
        model_cls = ViTPlanner
    planner = model_cls(config)
    checkpoint = torch.load(config.experiment.planner_checkpoint, map_location="cpu")
    planner.load_state_dict(checkpoint["model"])
    planner.eval()
    planner.requires_grad_(False)
    return planner

def get_titok_uni_reconstructor(config):
    if config.model.generator.model_type == "UViT":
        model_cls = UViTBert
    elif config.model.generator.model_type == "ViT":
        model_cls = ImageBert
    reconstructor = model_cls(config)
    checkpoint = torch.load(config.experiment.reconstructor_checkpoint, map_location="cpu")
    reconstructor.load_state_dict(checkpoint["model"])
    reconstructor.eval()
    reconstructor.requires_grad_(False)
    return reconstructor

# original sample function for MaskGit style sampling or discrete diffusion sampling
@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              conf_method="logit",
              randomize_temperature=2.0,
              logit_temperature=1.0,
              softmax_temperature_annealing=False,
              logit_temp_anneal=True,
              conf_anneal=True,
              num_sample_steps=8,
              device="cuda"):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        conf_method=conf_method,
        randomize_temperature=randomize_temperature,
        logit_temperature=logit_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        logit_temp_anneal=logit_temp_anneal,
        conf_anneal=conf_anneal,
        num_sample_steps=num_sample_steps)
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image

@torch.no_grad()
def ddpd_sample_fn(generator,
                planner,
                tokenizer,
                labels=None,
                guidance_scale=3.0,
                guidance_decay="constant",
                guidance_scale_pow=3.0,
                randomize_temperature=2.0,
                softmax_temperature_annealing=False,
                logit_temperature=1.0,
                conf_method="logit",
                conf_anneal=True,
                logit_temp_anneal=True,
                num_sample_steps=8,
                num_refinement_steps=0,
                refine_delta=5,
                conf_tol_min=-100,
                device="cuda"):
    generator.eval()
    tokenizer.eval()
    planner.eval()
    
    dpd = DDPD(generator, planner, tokenizer.config.model.vq_model.codebook_size, tokenizer.num_latent_tokens)
    
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = dpd.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        logit_temperature=logit_temperature,
        conf_method=conf_method,
        conf_anneal=conf_anneal,
        logit_temp_anneal=logit_temp_anneal,
        num_sample_steps=num_sample_steps,
        num_refinement_steps=num_refinement_steps,
        refine_delta=refine_delta,
        conf_tol_min=conf_tol_min,
        debug=False)
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image