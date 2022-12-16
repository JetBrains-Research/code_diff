import torch
import numpy as np


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs

def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion:
    def __init__(
        self, *, betas,
        rescale_timesteps=False,
        model_arch=None,
        training_mode='emb'
    ):
        self.rescale_timesteps = rescale_timesteps
        self.model_arch=model_arch

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.training_mode = training_mode
        self.mapping_func = None

    def training_losses(self, model, *args, **kwargs):
        return self.training_losses_e2e(model, *args, **kwargs)

    def calc_bpd_loop(self, model, *args, **kwargs):
        return self.calc_bpd_loop_e2e(model, *args, **kwargs)

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance2(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.size(0), x.size(-1)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        model_variance, model_log_variance = (
            np.append(self.posterior_variance[1], self.betas[1:]),
            np.log(np.append(self.posterior_variance[1], self.betas[1:])),
        )
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        print('should go here')
        pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.size(0), x.size(-1)
        
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_variance, model_log_variance = (
            np.append(self.posterior_variance[1], self.betas[1:]),
            np.log(np.append(self.posterior_variance[1], self.betas[1:])),
        )
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            noise = torch.randn_like(x)
            replace_mask = torch.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = torch.randn_like(noise[replace_mask])
                replace_mask = torch.abs(noise) > top_p
            assert (torch.abs(noise) <= top_p).all()

        else:
            noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out}

    def p_debug_loop(self,
                    model,
                    shape,
                    noise=None,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                    device=None,
                    progress=False,):
        final = None
        for sample in self.p_debug_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_debug_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            custom_t_start=100, 
    ):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(custom_t_start))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                img = out["sample"]

    def p_sample_loop_langevin_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        langevin_func=None,
        top_p=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                if langevin_func is not None:
                    out['t'] = t
                    out['img'] = img 
                    out = langevin_func(out)
                yield out
                img = out["sample"]


    def p_sample_loop_progressive_infill(
        self,
        model,
        shape,
        partial_enc,
        partial_mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        greedy=False
    ):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            t_batch = torch.tensor([self.num_timesteps - 1] * shape[0], device=device)
            partial_enc_with_noise = self.q_sample(partial_enc, t_batch)
            img = torch.randn(*shape, device=device)
            img[~partial_mask] = partial_enc_with_noise[~partial_mask]
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if i > 0:
                    partial_enc_with_noise = self.q_sample(partial_enc, t-1)
                else:
                    partial_enc_with_noise = partial_enc
                if greedy:
                    img = out["greedy_mean"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                else:
                    img = out["sample"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                yield out


    def p_sample_loop_progressive_merge(
        self,
        model,
        shape,
        partial_enc,
        partial_mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        greedy=False
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            t_batch = torch.tensor([self.num_timesteps - 1] * shape[0], device=device)
            partial_enc_with_noise = self.q_sample(partial_enc, t_batch)
            img = torch.randn(*shape, device=device)
            img[~partial_mask] = partial_enc_with_noise[~partial_mask]
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if i > 0:
                    partial_enc_with_noise = self.q_sample(partial_enc, t-1)
                else:
                    partial_enc_with_noise = partial_enc
                if greedy:
                    img = out["greedy_mean"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                else:
                    img = out["sample"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                yield out

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample=langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        top_p=-1.0,
        langevin_fn=None,
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            langevin_fn=langevin_fn,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    langevin_fn=langevin_fn,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None,
            noise=None, denoised_fn=None,
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        if model_kwargs is not None and 'input_ids' in model_kwargs:
            input_ids = model_kwargs.pop('input_ids')
            mapping_func = model_kwargs.pop('mapping_func', self.mapping_func)
        else:
            input_ids = None
            # noise=None
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        if input_ids is not None:
            assert mapping_func is not None 
            if mapping_func is not None and torch.any(t == 0):
                decoder_nll = mapping_func(out["mean"], input_ids) / out["mean"].size(-1)
            else:
                decoder_nll = torch.zeros_like(x_start)
            model_kwargs['input_ids'] = input_ids
            model_kwargs['mapping_func'] = mapping_func
        else:
            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
            )
            assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _vb_terms_bpd_e2e(
            self, model, x_start, x_t, t, input_ids, get_logits, x_start_mean, x_start_log_var, clip_denoised=True,
            model_kwargs=None, noise=None,denoised_fn=None,
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        model_kwargs.pop('mapping_func', self.mapping_func)

        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids) #t=-1

        decoder_nll = decoder_nll / out["mean"].size(-1)
        decoder_nll = decoder_nll / np.log(2.0)

        mask_1 = (t == 0)
        if mask_1.any():
            kl_T = normal_kl(
                x_start_mean, x_start_log_var, out["mean"], out["log_variance"]
            )
            kl_T = mean_flat(kl_T) / np.log(2.0)
            kl = torch.where(mask_1, kl_T, kl)

        out_mean, out_variance, \
        out_log_variance_clipped = self.q_mean_variance(x_start,
                                                        torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        kl_T = normal_kl(
            out_mean, out_log_variance_clipped, 0, 0
        )
        kl_T = mean_flat(kl_T) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = kl + decoder_nll + kl_T 
        return {"output": output, "pred_xstart": out["pred_xstart"], 'kl': kl, 'decoder_nll':decoder_nll, 'kl_T':kl_T}

    def get_x_start(self, x_start_mean, std):
        noise = torch.randn_like(x_start_mean)
        return x_start_mean + std * noise

    def token_discrete_loss(self, x_t, get_logits, input_ids):
        if self.model_arch == 'conv-unet' or  self.model_arch == '1d-unet':
            reshaped_x_t = x_t.view(x_t.size(0), x_t.size(1), -1).permute(0, 2, 1)
        else:
            reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        decoder_nll = decoder_nll.mean(dim=-1)
        return decoder_nll

    def x0_helper(self, model_output, x, t):
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        pred_prev, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}


    def training_losses_e2e(self, model, x_start, t, model_kwargs=None, noise=None):
        input_ids = model_kwargs.pop('input_ids').to(t.device)
        x_start_mean = model.model.get_embeds(input_ids)
        # x_start_mean = model.model.module.get_embeds(input_ids)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start_log_var = 2 * torch.log(std)
        x_start = self.get_x_start(x_start_mean, std)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # reparametrization trick.
        get_logits = model.model.get_logits
        # get_logits = model.model.module.get_logits

        terms = {}
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # B, C = x_t.size(0), x_t.size(-1)
        
        # print("From inside", model_output.size(), B, C, x_t.size())
        # model_output, model_var_values = torch.split(model_output, C, dim=-1)
        # frozen_out = torch.cat([model_output.detach(), model_var_values], dim=-1)

        # terms["vb"] = self._vb_terms_bpd_e2e(
        #     model=lambda *args, r=frozen_out: r,
        #     x_start=x_start,
        #     x_t=x_t,
        #     t=t,
        #     input_ids=input_ids,
        #     get_logits=get_logits,
        #     x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
        #     clip_denoised=False,
        #     noise=noise,
        # )["output"]
                        
        target = noise
        terms["mse"] = mean_flat((target - model_output) ** 2)
        model_out_x_start = self.x0_helper(model_output, x_t, t)['pred_xstart']
        t0_mask = (t == 0)
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        tT_loss =  mean_flat(out_mean ** 2)

        decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids)

        # if "vb" in terms:
        #     terms["loss"] = terms["mse"] + terms["vb"]
        # else:
        terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)
        
        return terms

    def _prior_bpd(self, x_start):
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop_e2e(self, model, x_start, clip_denoised=True, model_kwargs=None, denoised_fn=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        input_ids = model_kwargs.pop('input_ids').to(device)
        x_start_mean = model.get_embeds(input_ids)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start_log_var = 2 * torch.log(std)
        x_start = self.get_x_start(x_start_mean, std)
        get_logits = model.get_logits

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                out = self._vb_terms_bpd_e2e(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                )
            if t == self.num_timesteps -1:
                assert len(vb) == 0
                vb.append(out["kl_T"])
            vb.append(out["kl"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))
        vb.append(out["decoder_nll"])

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = out["kl_T"]
        total_bpd = vb.sum(dim=1)
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def calc_bpd_loop_emb(self, model, x_start, clip_denoised=True, model_kwargs=None,
                          denoised_fn=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
