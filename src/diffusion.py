import torch
import numpy as np


def _extract_into_tensor(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion:
    def __init__(self, betas, device=torch.device("cpu")):
        self.device = device
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

    def loss(self, model, x_start, t, input_ids):
        # NOISING
        # calculate Emb(w), Section 4.1
        x_start_mean = model.get_embeds(input_ids)
        # sigma_0 = sqrt(beta_0), Section 3.3
        # take q_phi(x_0 | w) = N(Emb(w), sigma_0^2 I), Section 4.1
        x_start = x_start_mean + np.sqrt(self.betas[0]) * torch.randn_like(x_start_mean)
        # q(x_t | x_t-1) = N(П sqrt(1-beta), 1 - П (1 - beta))
        # Node, there is not П beta, there is maybe a misprint in paper, but code is correct
        noise = torch.randn_like(x_start)
        x_t = _extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_start.shape, self.device) * x_start +\
              _extract_into_tensor(np.sqrt(1.0 - self.alphas_cumprod), t, x_start.shape, self.device) * noise       

        # DENOISING
        # p_theta (x_t-1 | x_t)
        # mu_theta(x_t, t)
        model_output = model(x_t, t)

        # L_simple(x_0) = ||eps_theta(x_t, t) - eps||^2
        mse = mean_flat((noise - model_output) ** 2)
        # f_theta(x_t, t), Section 4.2 and footnote 4 page 5
        model_out_x_start = _extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod), t, x_t.shape, self.device) * x_t -\
                            _extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod - 1), t, x_t.shape, self.device) * model_output
        # ||f_theta(x_t, t) - x_0||^2, Section 4.2
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
        mse = torch.where(t == 0, t0_loss, mse)

        # q(x_T | x_0) from (1)
        # ||sqrt(П (1 - beta)) x_0||^2
        last_tic_loss =  mean_flat((_extract_into_tensor(np.sqrt(self.alphas_cumprod), self.betas.shape[0] - 1, x_start.shape, self.device) * x_start) ** 2)

        # cross entropy H(x_0, w) = -E_q log p = -log p_theta(w | x_0)
        decoder_nll = self.token_discrete_loss(x_start, model.get_logits, input_ids)
        
        return mse + decoder_nll + last_tic_loss
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_start.shape, self.device) * x_start
            + _extract_into_tensor(np.sqrt(1.0 - self.alphas_cumprod), t, x_start.shape, self.device)
            * noise
        )

    def sample(self, model, shape, rounding):
        sample = torch.randn(*shape, device=self.device)
        for i in range(self.num_timesteps, 0, -1):
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                sample = self.p_sample(model, sample, t, rounding)
        return sample

    def p_sample(self, model, x, t, rounding):
        out = self.p_mean_variance(model, x, t, rounding)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return sample

    def p_mean_variance(self, model, x, t, rounding):
        # p_theta (x_t-1 | x_t)
        # mu_theta(x_t, t)
        model_output = model(x, t)

        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        model_variance = np.append(posterior_variance[1], self.betas[1:])
        model_log_variance = _extract_into_tensor(np.log(model_variance), t, x.shape, self.device)

        # f_theta(x_t, t), Section 4.2 and footnote 4 page 5
        x_start = _extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod), t, x.shape, self.device) * x -\
                  _extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod - 1), t, x.shape, self.device) * model_output
        # rounding
        x_start = rounding(x_start, t)
            
        model_mean = self.q_posterior_mean(x_start=x_start, x_t=x, t=t)
        return {
            "mean": model_mean,
            "log_variance": model_log_variance
        }

    def q_posterior_mean(self, x_start, x_t, t):
        A = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        B = (1.0 - self.alphas_cumprod_prev) * np.sqrt(1 - self.betas) / (1.0 - self.alphas_cumprod)
        return _extract_into_tensor(A, t, x_t.shape, self.device) * x_start +\
               _extract_into_tensor(B, t, x_t.shape, self.device) * x_t

    def token_discrete_loss(self, x_t, get_logits, input_ids):
        logits = get_logits(x_t)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.flatten()).view(input_ids.shape)
        return decoder_nll.mean(dim=-1)
