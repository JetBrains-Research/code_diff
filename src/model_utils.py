import torch
import numpy as np
from torch import nn
from functools import partial
from .gaussian_diffusion import GaussianDiffusion


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, diffusion_steps, model, device, **kwargs):
        self.use_timesteps = set(space_timesteps(diffusion_steps, [diffusion_steps]))
        self.timestep_map = []

        alpha_bar = lambda t: 1 - np.sqrt(t + 0.0001)
        max_beta = 0.999
        betas = []
        for i in range(diffusion_steps):
            t1 = i / diffusion_steps
            t2 = (i + 1) / diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        kwargs["betas"] = np.array(betas)
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0

        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)
        self.mapping_func = partial(compute_logp, model, device)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class UniformSampler():
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights
    
    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


def compute_logp(original_model, device, x, input_ids):
    input_embs = original_model.transformer.wte
    down_proj = original_model.down_proj
    down_proj_emb = down_proj(input_embs.weight)
    model = nn.Embedding(down_proj_emb.size(0), down_proj_emb.size(1))
    model.weight.data = down_proj_emb
    model.weight.requires_grad = False
    
    word_emb = model.weight.to(device)
    sigma = 0.1
    batch_size, seqlen, _ = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)
    word_emb_flat = word_emb.unsqueeze(1)
    diff = (x_flat - word_emb_flat) ** 2

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)
    logp_expanded = logp_expanded.permute((1, 0))
    
    cross_entropy = nn.CrossEntropyLoss(reduction='none')
    return cross_entropy(logp_expanded, input_ids.view(-1)).view(batch_size, seqlen)


def space_timesteps(num_timesteps, section_counts):
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
