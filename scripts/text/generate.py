import os
import json
import torch
import numpy as np
from functools import partial
from src import Transformer, GaussianDiffusion
from transformers import AutoTokenizer


def get_weights(model):
    # input_embs = model.transformer.wte
    # down_proj = model.down_proj
    # down_proj_emb = down_proj(input_embs.weight)
    # model = torch.nn.Embedding(down_proj_emb.size(0), down_proj_emb.size(1))
    # model.weight.data = down_proj_emb    
    model.weight.requires_grad = False
    return model

def rounding(model, text_emb, t):
    down_proj_emb = model.weight
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb):
        emb_norm = (down_proj_emb**2).sum(-1).view(-1, 1) #vocab
        text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) #d, bsz*seqlen
        arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) #bsz*seqlen, 1
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb, text_emb_t) #(vocab, d) x (d, bsz*seqlen)
        dist = torch.clamp(dist, 0.0, np.inf)
        return torch.topk(-dist, k=1, dim=0).indices

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    indices = get_efficient_knn(down_proj_emb, text_emb.to(down_proj_emb.device))
    rounded_tokens = indices[0]
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    return new_embeds

device = torch.device('cuda:3')
# device = torch.device('cpu')

channels = 8
batch_size = 32
sigma_small = True
num_samples = 32 # 1024
num_samples = (num_samples // batch_size) * batch_size
diffusion_steps = 1000

channel_mult = (1, 2, 3, 4)
attention_ds = []
for res in [16, 8]:
    attention_ds.append(64 // int(res))

rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

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
)
model.load_state_dict(torch.load("models/text/diffusion_lm.model"))
model.to(device)
model.eval()

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

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model2 = torch.nn.Embedding(len(tokenizer), channels)
model2.weight = torch.nn.Parameter(model.word_embedding.weight.clone().cpu())    
model3 = get_weights(model2)

# seq_len = 64
seq_len = 10
x_t = torch.zeros((num_samples, seq_len, channels)).to(device)

for i in range(num_samples // batch_size):
    sample = diffusion.sample(
        model, (batch_size, seq_len, channels), partial(rounding, model3.cuda()))
    x_t[i * batch_size: (i + 1) * batch_size] = sample

logits = model.get_logits(x_t)
tokens = torch.topk(logits, k=1, dim=-1).indices

# with open('data/e2e_data/vocab.json', 'r') as f:
#     vocab = json.load(f)
# tokenizer = {v: k for k, v in vocab.items()}

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

os.makedirs("outputs", exist_ok=True)
file = open("outputs/lm.txt", "a")

words = []
for seq in tokens:
    words.append(" ".join([tokenizer.convert_ids_to_tokens(x.item()) for x in seq]))
    file.write(words[-1] + "\n")

file.close()
