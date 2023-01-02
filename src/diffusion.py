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
    def __init__(self, betas, device):
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
        # q(x_t | x_t-1) = N(П sqrt(1-beta), П beta)
        # TODO: but here there is not П beta, but 1 - П (1 - beta)
        noise = torch.randn_like(x_start)
        x_t = _extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_start.shape, self.device) * x_start +\
              _extract_into_tensor(np.sqrt(1.0 - self.alphas_cumprod), t, x_start.shape, self.device) * noise       

        # DENOISING
        # p_theta (x_t-1 | x_t)
        # mu_theta(x_t, t)
        model_output = model(x_t, t)

        # L_simple(x_0) = ||mu_theta(x_t, t) - TODO: ?||^2
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

    def token_discrete_loss(self, x_t, get_logits, input_ids):
        logits = get_logits(x_t)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.flatten()).view(input_ids.shape)
        return decoder_nll.mean(dim=-1)
