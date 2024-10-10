torchrun --standalone --nproc_per_node=4 train_denoiser.py text8/config/train_denoiser.py --batch_size=512 \
  --gradient_accumulation_steps=4 --resume_dir=None --wandb_run_name='ddpd_denoiser_mask' \
  --model_type='ddpd_denoiser_mask'
