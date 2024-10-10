out_dir = '/pscratch/sd/s/sulinl/ddpd/out-text8'
eval_interval = 2000
eval_iters = 200
log_interval = 1

data_dir = '/pscratch/sd/s/sulinl/dfm/data/text8'

warm_start_ckpt = None
init_from = 'scratch'
resume_dir = None
always_save_checkpoint = False

wandb_log = True
wandb_project = 'text8'
wandb_run_name = 'ddpd_denoiser'
wandb_id = 'blank'
is_repeat = False
wandb_mode = 'online'

dataset = 'text8'
gradient_accumulation_steps = 4
batch_size = 512
block_size = 256
overfit_batch = False

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
qk_layernorm = True

model_type = 'ddpd_denoiser_mask' # or 'ddpd_denoiser_uniform'

learning_rate = 1e-4
max_iters = 750000
lr_decay_iters = 1000000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 1000
min_t = 0.0

