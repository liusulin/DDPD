
out_dir = '/pscratch/sd/s/sulinl/ddpd'
out_folder = 'out-text8-samples'
planner_ckpt_path = '/pscratch/sd/s/sulinl/ddpd_pretrained/text8/planner_model.pt'
# '/pscratch/sd/s/sulinl/dpd/out-text8/2024-04-27-6w2cgc92_dpd_v3_no_te_pred_mask/ckpt_450000.pt'
denoiser_ckpt_path = '/pscratch/sd/s/sulinl/ddpd_pretrained/text8/denoiser_model_uniformD.pt'
# '/pscratch/sd/s/sulinl/ddpd_pretrained/text8/denoiser_model_maskD.pt'
# '/pscratch/sd/s/sulinl/dfm/out-text8/2024-04-21-18-02-28_dfm_nosc/best_ckpt.pt'
# '/pscratch/sd/s/sulinl/dfm/out-text8/2024-04-23-gklqyh74_dfm_uniform_nosc/best_ckpt.pt'

data_dir = '/pscratch/sd/s/sulinl/dfm/data/text8'
is_mask_denoiser = False
run_name = 'base'

dataset = 'text8'
batch_size = 512
block_size = 256 # context of up to 256 previous characters

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
qk_layernorm = True
do_x1_sc = False

total_samples = 512
timesteps = 1000
max_t = 0.98
argmax_final = True
noise = 0.0
x1_temp = 1.0

# do_purity_sampling = False
# purity_temp = 1.0

model_type = 'ddpd'