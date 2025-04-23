import math
import torch
import torchvision.utils as vutils
import torch.nn.functional as F

import wandb

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.lr * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.lr - config.min_lr)

    
def log_image_and_pred_D(tokenizer, code, noisy_code, mask, samples, step, n_sample=10, fixed_time=False, num_tokens=128):
    """ Log the image and the prediction of the model"""
    log_label_name = "Images" if not fixed_time else "Images_fixed_time"
    accuracy_label_name = "Accuracy" if not fixed_time else "Accuracy_fixed_time"
    mask_rate_label_name = "Mask_rate" if not fixed_time else "Mask_rate_fixed_time"
    predicted_code = torch.where(mask, samples, noisy_code)
    with torch.no_grad():
        all_samples = torch.cat([code[:n_sample], noisy_code[:n_sample], predicted_code[:n_sample]], dim=0)
        all_samples = tokenizer.decode_tokens(all_samples)
        # split the samples into original, noisy and predicted
        orig_sample = all_samples[:n_sample]
        noisy_sample = all_samples[n_sample:2*n_sample]
        predicted_sample = all_samples[2*n_sample:]
    sample_all = torch.cat([orig_sample, noisy_sample, predicted_sample], dim=0)
    all_grid = vutils.make_grid(sample_all, nrow=n_sample, padding=2, normalize=False)
    wandb.log({f"{log_label_name}/Image": [wandb.Image(all_grid, caption="Original, noisy and predicted images")]}, step=step)
    
    matches = (samples == code).float()
    acc = (matches * mask.float()).sum() / mask.sum()
    # average masking rate
    mask_rate = mask.float().mean()
    wandb.log({f"Eval/{accuracy_label_name}": acc}, step=step)
    wandb.log({f"Eval/{mask_rate_label_name}": mask_rate}, step=step)
    # plot the original mask
    nrow = 8
    ncol = num_tokens // nrow
    mask_sample = mask[:n_sample].float().view(n_sample, nrow, ncol).unsqueeze(1)
    # resize it from 16 to 256 or 512 by extending 1 pixel to 16 pixels
    mask_sample = F.interpolate(mask_sample, (nrow*16, ncol*16))
    mask_grid = vutils.make_grid(mask_sample, nrow=n_sample, padding=2, normalize=False)
    wandb.log({f"{log_label_name}/Mask": [wandb.Image(mask_grid, caption="Original mask")]}, step=step)

def log_image_and_pred(tokenizer, code, noisy_code, mask, pred_prob, step, n_sample=10, fixed_time=False, num_tokens=128):
    """ Log the image and the prediction of the model"""
    log_label_name = "Images" if not fixed_time else "Images_fixed_time"
    accuracy_label_name = "Accuracy" if not fixed_time else "Accuracy_fixed_time"
    mask_rate_label_name = "Mask_rate" if not fixed_time else "Mask_rate_fixed_time"
    with torch.no_grad():
        orig_sample = tokenizer.decode_tokens(code[:n_sample])
        
    # save the masked image
    with torch.no_grad():
        noisy_sample = tokenizer.decode_tokens(noisy_code[:n_sample])
    sample_all = torch.cat([orig_sample, noisy_sample], dim=0)
    all_grid = vutils.make_grid(sample_all, nrow=n_sample, padding=2, normalize=False)
    wandb.log({f"{log_label_name}/Image": [wandb.Image(all_grid, caption="Original and noisy images")]}, step=step)
    
    # sample from pred_prob and calulate the accruacy
    is_mask_pred = torch.bernoulli(pred_prob)
    acc = (is_mask_pred.bool() == mask).float().mean()
    # average masking rate
    mask_rate = mask.float().mean()
    wandb.log({f"Eval/{accuracy_label_name}": acc}, step=step)
    wandb.log({f"Eval/{mask_rate_label_name}": mask_rate}, step=step)
    # plot the original mask
    nrow = 8
    ncol = num_tokens // nrow
    mask_sample = mask[:n_sample].float().view(n_sample, nrow, ncol).unsqueeze(1)
    # resize it from 16 to 256 or 512 by extending 1 pixel to 16 pixels
    mask_sample = F.interpolate(mask_sample, (nrow*16, ncol*16))
    pred_prob = pred_prob[:n_sample].view(n_sample, nrow, ncol).unsqueeze(1)
    pred_prob = F.interpolate(pred_prob, (nrow*16, ncol*16))
    
    mask_all = torch.cat([mask_sample, pred_prob], dim=0)
    mask_grid = vutils.make_grid(mask_all, nrow=n_sample, padding=2, normalize=False)
    wandb.log({f"{log_label_name}/Mask": [wandb.Image(mask_grid, caption="Original mask and predicted mask")]}, step=step)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count