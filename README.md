# DDPD: Discrete Diffusion with Planned Denoising
Code repository for the paper [Think While You Generate: Discrete Diffusion with Planned Denoising](https://arxiv.org/abs/2410.06264),
by Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi Jaakkola, Rafael Gómez-Bombarelli.

![DDPD overview](./assets/ddpd_main.png)

**Sampling process of DDPD:**
A planner is first used to determine which positions are most likely to be noise and should be denoised next. The denoiser is then applied to the selected positions conditioned on all the input tokens.
![Sampling process](./assets/sampling_process.png)

**Training objectives of the planner and denoiser:**
Cross-entropy loss for predicting the binary mask of noise tokens for the planner and cross-entropy loss for predicting the original token values for the denoiser.
![Training objectives](./assets/training_objectives.png)




--- 
## Code for text8 language modeling task
### Install
Package requirements are listed in `ddpd_text.yml`. [Mamba](https://mamba.readthedocs.io/en/latest/) is recommended for faster installation.

```shell
conda env create -f ddpd_text.yml
```

### Pretrained models
Our pretrained models can be downloaed at [this link](https://huggingface.co/sulinliu/ddpd/tree/main).

### Run training: denoiser
```python
torchrun --standalone --nproc_per_node=4 train_denoiser.py text8/config/train_denoiser.py --batch_size=512 --gradient_accumulation_steps=4 --resume_dir=None --wandb_run_name='ddpd_denoiser_mask' --model_type='ddpd_denoiser_mask'
```
On a single node with 4 GPUs of 80GB memory, `batch_size` and `gradient_accumulation_steps` can be adjusted to fit in smaller GPU memory.
### Run training: planner
```python
torchrun --standalone --nproc_per_node=4 train_planner.py text8/config/train_planner.py --batch_size=512 --gradient_accumulation_steps=4 --resume_dir=None --wandb_run_name='ddpd_planner' --model_type='ddpd_planner'
```

### Run sampling code
```python
python sample_text8.py text8/config/sample.py
```
To replicate the results of DDPD-MaskD and DDPD-UniD in the following figure,  
```shell
bash text8/scripts/generate_samples_ddpd_maskD.sh
bash text8/scripts/generate_samples_ddpd_uniD.sh
bash text8/scripts/evaluate_samples_ddpd_maskD.sh
bash text8/scripts/evaluate_samples_ddpd_uniD.sh
```
**Results on text8 conditional generation task:**
![Language modeling results](./assets/text8.png)

## Code for OpenWebText language modeling task (Coming soon)

**Results on OpenWebText language modeling unconditional generation task:**
![Language modeling results](./assets/owt.png)


## Code for ImageNet 256x256 token generation task (Coming soon)

**Results on ImageNet 256x256 token generation task:**
![ImageNet 256x256 token generation results](./assets/imagenet_256.png)

## Citation
```
@article{liu2024ddpd,
  title={Think While You Generate: Discrete Diffusion with Planned Denoising},
  author={Liu, Sulin and Nam, Juno and Campbell, Andrew and Stärk, Hannes and Xu, Yilun and Jaakkola, Tommi and Gómez-Bombarelli, Rafael},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```