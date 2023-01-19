import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
from src import load_data_text
from transformers import AutoTokenizer
from src import GaussianDiffusion, UniformSampler, Transformer

use_wandb = False
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
alpha_bar = lambda t: 1 - np.sqrt(t + 1e-16)
max_beta = 1 - 1e-3
betas = []
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

diffusion = GaussianDiffusion(betas, device)
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

optimizer = torch.optim.Adam(model.parameters())
epochs = 5
validation_every = 250
if use_wandb:
    wandb.init(project="diffusion-lm")

print("Start training...")
for epoch in range(epochs):
    train_batches = 0
    train_loss = 0
    print('Epoch', epoch + 1, '/', epochs)
    for i, (batch, input_ids) in enumerate(tqdm(data)):  
        optimizer.zero_grad()
        batch = batch.to(device)
        input_ids = input_ids.to(device)
        t, weights = schedule_sampler.sample(batch.shape[0], device)

        true_loss = diffusion.loss(model, batch, t, input_ids)
        train_loss += true_loss.mean()
        train_batches += 1
        loss = (true_loss * weights).mean()
        loss.backward()
        optimizer.step()
        if use_wandb:
            wandb.log({"train/loss": loss})

        if (i + 1) % validation_every == 0:
            val_batches = 0
            val_loss = 0
            print('Validation')
            with torch.no_grad():
                for (batch_eval, input_ids_eval) in tqdm(data_valid):
                    batch_eval_eval = batch_eval.to(device)
                    input_ids_eval = input_ids_eval.to(device)
                    t, weights = schedule_sampler.sample(batch_eval.shape[0], device)

                    val_loss += diffusion.loss(model, batch_eval, t, input_ids_eval).mean()
                    val_batches += 1
            
            val_loss = val_loss.cpu()
            if use_wandb:
                wandb.log({"val/loss": val_loss / val_batches})
            print('validation loss =', val_loss / val_batches)

    train_loss = train_loss.cpu()
    if use_wandb:
        wandb.log({"train_epoch/loss": train_loss / train_batches})
    print('train epoch loss =', train_loss / train_batches)
