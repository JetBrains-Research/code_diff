import os
import copy
import wandb
import torch
import numpy as np
from tqdm import tqdm
from src import load_data_text
from transformers import AutoTokenizer
from src import GaussianDiffusion, UniformSampler, Transformer

device = torch.device('cuda:3')

channel_mult = (1, 2, 3, 4)
attention_ds = []
for res in [16, 8]:
    attention_ds.append(64 // int(res))

print("Creating models...")

rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
channels = 8
model2 = torch.nn.Embedding(len(rev_tokenizer), channels)
path_save = 'data/e2e_data/random_emb.torch'
if not os.path.exists(path_save):
    torch.save(model2.state_dict(), path_save)
print('vocab size:', len(rev_tokenizer))

model = Transformer(
    in_channels=channels, out_channels=channels,
    model_channels=128,
    num_res_blocks=2,
    attention_resolutions=tuple(attention_ds),
    channel_mult=channel_mult,
    num_classes=None,
    num_heads=4,
    vocab_size=len(rev_tokenizer),
    logits_mode=1,
).to(device)

diffusion_steps = 1000

# Take sqrt noise schedule alpha_bar from [Xiang Lisa Li et al., 2022], Appendix A
# Calculate beta_t by factorizing their product alpha_bar_t (be definition, Section 4.2)
alpha_bar = lambda t: 1 - np.sqrt(t + 0.0001)
max_beta = 0.999
betas = []
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

diffusion = GaussianDiffusion(betas, model, device)
schedule_sampler = UniformSampler(diffusion)

print("Loading train dataset...")
data = load_data_text(
    batch_size=32,
    split='train',
    vocab_dict=rev_tokenizer,
    model=model2
)

data_valid = load_data_text(
    batch_size=32,
    split='valid',
    vocab_dict=rev_tokenizer,
    model=model2
)

ema_rate = [0.9999]

model_params = list(model.parameters())
master_params = model_params

optimizer = torch.optim.Adam(master_params)
ema_params = [copy.deepcopy(master_params) for _ in range(len(ema_rate))]

epochs = 5
validation_every = 250
wandb.init(project="diffusion-lm", id="refactoring3")

print("Start training...")
for epoch in range(epochs):
    train_batches = 0
    train_loss = 0
    print('Epoch', epoch + 1, '/', epochs)
    for i, (batch, cond) in enumerate(tqdm(data)):  
        for p in model_params:
            if p.grad is not None:
                p.grad.zero_()

        batch = batch.to(device)
        micro_cond = {k: v.to(device) for k, v in cond.items()}
        t, weights = schedule_sampler.sample(batch.shape[0], device)

        true_loss = diffusion.training_losses(model, batch, t, model_kwargs=micro_cond)["loss"]
        train_loss += true_loss.mean()
        train_batches += 1
        loss = (true_loss * weights).mean()
        loss.backward()
        optimizer.step()
        wandb.log({"train/loss": loss})

        for rate, params in zip(ema_rate, ema_params):
            for targ, src in zip(params, master_params):
                targ.detach().mul_(rate).add_(src, alpha=1 - rate)

        if (i + 1) % validation_every == 0:
            val_batches = 0
            val_loss = 0
            print('Validation')
            with torch.no_grad():
                for (batch_eval, cond_eval) in tqdm(data_valid):
                    batch_eval = batch_eval.to(device)
                    micro_cond = {k: v.to(device) for k, v in cond_eval.items()}
                    t, weights = schedule_sampler.sample(batch_eval.shape[0], device)

                    val_loss += diffusion.training_losses(model, batch_eval, t, model_kwargs=micro_cond)["loss"].mean()
                    val_batches += 1
            
            val_loss = val_loss.cpu()
            wandb.log({"val/loss": val_loss / val_batches})
            print('validation loss =', val_loss / val_batches)

    train_loss = train_loss.cpu()
    wandb.log({"train_epoch/loss": train_loss / train_batches})
    print('train epoch loss =', train_loss / train_batches)
