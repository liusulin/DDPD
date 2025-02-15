import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}

# Function for top-p (nucleus) sampling
def top_p_sampling(probs_x0, top_p=0.8):
    sorted_probs, sorted_indices = torch.sort(probs_x0, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = (cumulative_probs > top_p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter the mask back to the original shape
    indices_to_remove = torch.zeros_like(probs_x0, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    probs_x0 = probs_x0.masked_fill(indices_to_remove, 0.0)
    return probs_x0 / probs_x0.sum(-1, keepdim=True)

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            if i % 200 == 0:
                print(f"step {i} done")

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

def get_ddpd_sampler(mask_graph, uniform_graph, noise, batch_dims, predictor, steps, top_p=1.0,
                        denoise=True, eps=1e-5, use_prob_for_dim_change=False, device=torch.device('cpu'), 
                        proj_fun=lambda x: x):
    projector = proj_fun
    predictor = get_predictor(predictor)(mask_graph, noise)
    denoiser = Denoiser(mask_graph, noise)

    @torch.no_grad()
    def ddpd_sampler(model, pred_noise_model):
        pred_noise_model_fn = mutils.get_model_fn(pred_noise_model, train=False)
        model_fn = mutils.get_model_fn(model, train=False) # to get denoiser model
        x = uniform_graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)        
        
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            sigma = torch.zeros(x.shape[0], device=device)
            log_if_noise = pred_noise_model_fn(x,sigma)
            # sample the dimension to change
            if not use_prob_for_dim_change: # if using softmax(logits) or original p(logits) for weighting dim change
                prob_change_d = F.softmax(log_if_noise.squeeze(-1), dim=-1)
            else:
                prob_change_d = torch.sigmoid(log_if_noise.squeeze(-1)) # (B, D)
            dim_change = sample_categorical(prob_change_d) # (B,)
            prob_noise = F.sigmoid(log_if_noise.squeeze(-1)) # (B, D)
            mask = torch.bernoulli(prob_noise).bool().long() # (B, D)
            mask[torch.arange(x.shape[0]), dim_change] = 1
            x_input = x * (1 - mask) + (mask_graph.dim - 1) * mask
            total_mask = mask.sum(-1, keepdim=True) 
            t = total_mask/1024 # must make sure t is of dim of (B, 1), since we squeeze it later
            if i % 200 == 0:
                print(f"step {i} done")
                print(t.squeeze())
            sigma = noise(t)[0].squeeze(-1)
            # get denoiser model score and transform it to p(x_1|x_t) probabilities
            probs_x0 = model_fn(x_input, sigma).exp()
            probs_x0 = torch.scatter(probs_x0, -1, x_input[..., None], torch.zeros_like(probs_x0[..., :1])) # setting the output at the indices to 0
            probs_x0 = probs_x0/probs_x0.sum(-1,keepdim=True) # normalize
            # Apply top-p sampling
            probs_x0 = top_p_sampling(probs_x0, top_p=top_p)
            sampled_values = sample_categorical(probs_x0)
            x[torch.arange(x.shape[0]), dim_change] = sampled_values[torch.arange(x.shape[0]), dim_change]
            
        if denoise:
            # final step to denoise in case something remains incorrect
            log_if_noise = pred_noise_model_fn(x,sigma)
            prob_noise = F.sigmoid(log_if_noise.squeeze(-1)) # (B, D)
            mask = torch.bernoulli(prob_noise).bool().long() # (B, D)
            x_input = x * (1 - mask) + (mask_graph.dim - 1) * mask
            total_mask = mask.sum(-1, keepdim=True) 
            print("Final step of denoising (optional): ")
            print(total_mask.squeeze())
            t = total_mask/1024 # must make sure t is of dim of (B, 1), since we squeeze it later
            sigma = noise(t)[0].squeeze(-1)
            probs_x0 = model_fn(x_input, sigma).exp()
            probs_x0 = torch.scatter(probs_x0, -1, x_input[..., None], torch.zeros_like(probs_x0[..., :1])) # setting the output at the indices to 0
            probs_x0 = probs_x0/probs_x0.sum(-1,keepdim=True) # normalize
            sampled_values = sample_categorical(probs_x0)
            x = x * (1 - mask) + sampled_values * mask
            
        return x
    
    return ddpd_sampler