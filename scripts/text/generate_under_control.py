import os
import json
import torch
import benepar
import numpy as np
from functools import partial
from spacy.lang.en import English
from src import GaussianDiffusion, Transformer, TreeControl
from improved_diffusion.rounding import load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round


def langevin_fn_tree(coeff, model_control, model3, label_ids, step_size, sample, mean, sigma, alpha, t, prev_sample):
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    input_embs_param = torch.nn.Parameter(sample)

    with torch.enable_grad():
        for _ in range(K):
            optimizer = torch.optim.Adam([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            
            model_out = model_control(input_embs=input_embs_param, parse_chart=label_ids, t=tt)
            coef = coeff

            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()

            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = torch.randn_like(input_embs_param.data)
            input_embs_param = torch.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())

    return input_embs_param.data


defaults = dict(
    data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
    out_dir="diffusion_lm/improved_diffusion/out_gen",
    emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
    partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
    start_idx=0, end_idx=0,
)
defaults.update(model_and_diffusion_defaults())

with open(config_path, 'rb', ) as f:
    training_args = json.load(f)

noise_level = 0.0
sigma_small = True

device = torch.device('cuda:3')

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

model = torch.load("models/text/diffusion_lm.model")
model.to(device)
model.eval()

model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
        os.path.split(args.model_path)[0])
    
model_embs.weight = torch.nn.Parameter(model.word_embedding.weight.clone().cpu())
model_embs = model_embs.cuda()
model3 = get_weights(model_embs, args)

if args.partial_seq:
    partial_seq = [args.partial_seq]
    partial_seq_idx = ['0']
elif args.partial_seq_file:
    # implies that we should read from the files
    nlp = English()
    tokenizer_spacy = nlp.tokenizer
    with open(args.partial_seq_file, 'r') as f:
        sent_lst = json.load(f)
    partial_seq = []
    partial_seq_idx = []
    for idx, (key, val) in enumerate(sent_lst.items()):
        if idx < int(args.start_idx) or idx > int(args.end_idx):
            continue
        partial_seq_ = f"{val['obs1']} " + "PAD " * 10 + f"{val['obs2']}"
        word_lst = [x.text for x in tokenizer_spacy(partial_seq_)]
        partial_seq_ = " ".join(word_lst)
        print(partial_seq_, idx)
        partial_seq.append(partial_seq_)
        partial_seq_idx.append(str(idx))
else:
    partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                   'Alimentum , situated by the river , is quite child friendly .']
    partial_seq_idx = ['0', '1']
    
tokens2id = {v:k for k, v in tokenizer.items()}
todo_pad_token = -1
pad_token = tokens2id['PAD']
encoded_partial_seq = [torch.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in partial_seq]
        
right_pad = torch.empty(64).fill_(pad_token).long()
encoded_partial_seq = [torch.cat([right_pad], dim=0)]
encoded_partial_seq[0][0] = tokens2id['START']
encoded_partial_seq[0][args.tgt_len] = tokens2id['END']
            
model_control = torch.load("models/text/control.model")
model_control.to(device)
model_control.eval()

parser = benepar.Parser("benepar_en3")
tree_vocab = parser._parser.config["label_vocab"]
tree_vocab_rev = {v: k for k, v in tree_vocab.items()}

control_label_lst = []
with open('diffusion_lm/improved-diffusion/control_gen/target_tree.json', 'r') as controlf:
    for line in controlf:
        control_label_lst.append(json.loads(line))
# print(control_label_lst[:1])
control_constraints = []
for label_class_dict in control_label_lst[100:]:
    padded_chart = torch.LongTensor(label_class_dict['padded_chart'])
    words_ = label_class_dict['words_']
    label_ids = padded_chart
    langevin_fn_selected = partial(langevin_fn_tree, 0.0005, model_control, model3.cuda(),
                                       label_ids.expand(args.batch_size, -1, -1),
                                        0.1)
    control_constraints.append((langevin_fn_selected, [label_class_dict['tree']]))

partial_seq = control_constraints
encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]

sample_dict = {}
    
for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq) :
    all_images = []
    all_labels = []
        
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size,-1)
        partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
            
        encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)

        encoded_seq_hidden = model_embs(encoded_seq.cuda())
        seqlen = encoded_seq.size(1)
                
        sample_shape = (args.batch_size, seqlen, args.in_channel, )

        langevin_fn_selected, label_class_attributes = control_helper
        loop_func_ = diffusion.ddim_sample_loop_progressive

        sample = loop_func_(
            model,
            sample_shape,
            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=encoded_seq_hidden.device,
            langevin_fn=langevin_fn_selected,
            eta=args.eta
        )[-1]["sample"]
    
        gathered_samples = [sample]
        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            all_labels.extend([classes.cpu().numpy()])
    
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    sample_dict[tuple(label_class_attributes)] = arr

if args.class_cond:
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
shape_str = "x".join([str(x) for x in arr.shape])
model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + '.' + str(os.path.split(args.model_path)[1])

def decode_helper(args, sample_dict, diff_model=None):
    result_dict = {}
        
    for k, v in sample_dict.items():
        arr = v
        word_lst_e2e = []
        x_t = torch.tensor(arr).cuda()
        reshaped_x_t = x_t
        logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = torch.topk(logits, k=1, dim=-1)
        tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
        for seq in cands.indices:
            tokens = " ".join([tokenizer[x[0].item()] for x in seq])
            word_lst_e2e.append(tokens)
        word_lst = word_lst_e2e
            
        result_dict[k] = word_lst
    return result_dict

out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.json")
fout = open(out_path_pipe, 'w')
result_dict = decode_helper(args, sample_dict, diff_model=model)
for k, word_lst in result_dict.items():
    print({k:word_lst}, file=fout)
fout.close()
out_path2 = out_path_pipe
