import os
import json
import copy
import torch
import catalyst
from src import load_data_text
from transformers import AutoTokenizer
from src import SpacedDiffusion, UniformSampler, Transformer

device = torch.device('cpu')

channel_mult = (1, 2, 3, 4)
attention_ds = []
for res in [16, 8]:
    attention_ds.append(64 // int(res))

print("Creating models...")
model = Transformer(
    in_channels=8, out_channels=8,
    model_channels=128,
    num_res_blocks=2,
    attention_resolutions=tuple(attention_ds),
    channel_mult=channel_mult,
    num_classes=None,
    num_heads=4,
    vocab_size=1000,
    logits_mode=1,
).to(device)

diffusion = SpacedDiffusion(
    diffusion_steps=1000,
    rescale_timesteps=True,
    model_arch='transformer',
    training_mode='e2e',
    device=device,
    model=model
)

schedule_sampler = UniformSampler(diffusion)

rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model2 = torch.nn.Embedding(len(rev_tokenizer), 16)

print("Loading train dataset...")
data = load_data_text(
    batch_size=32,
    load_vocab=None,
    split='train',
    model=None
)

print("Loading validation dataset...")
data_valid = load_data_text(
    batch_size=32,
    split='valid',
    load_vocab=rev_tokenizer,
    model=model2
)

ema_rate = [0.9999]

model_params = list(model.parameters())
master_params = model_params

optimizer = torch.optim.Adam(master_params)
ema_params = [copy.deepcopy(master_params) for _ in range(len(ema_rate))]

print("Start training...")
for i, (batch, cond) in enumerate(data):  
    for p in model_params:
        if p.grad is not None:
            p.grad.zero_()

    batch = batch.to(device)
    micro_cond = {k: v.to(device) for k, v in cond.items()}
    t, weights = schedule_sampler.sample(batch.shape[0], device)

    loss = (diffusion.training_losses(model, batch, t, model_kwargs=micro_cond)["loss"] * weights).mean()
    loss.backward()
    optimizer.step()
    print(loss)

    for rate, params in zip(ema_rate, ema_params):
        for targ, src in zip(params, master_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)

    if (i + 1) % 10 == 0:
        with torch.no_grad():
            for (batch_eval, cond_eval) in data_valid:
                for p in model_params:
                    if p.grad is not None:
                        p.grad.zero_()

                batch_eval = batch_eval.to(device)
                micro_cond = {k: v.to(device) for k, v in cond_eval.items()}
                t, weights = schedule_sampler.sample(batch_eval.shape[0], device)

                diffusion.training_losses(model, batch_eval, t, model_kwargs=micro_cond)
