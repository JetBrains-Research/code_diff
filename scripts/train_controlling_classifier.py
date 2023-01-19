import torch
import wandb
import benepar
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import Counter
from spacy.lang.en import English
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertConfig, default_data_collator
from src import GaussianDiffusion, TreeControl, chart_from_tree, pad_charts

# torch.multiprocessing.set_start_method('spawn', force=True)
benepar.download('benepar_en3')

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False, pad_mask_id=None):
    if pad_mask_id is None:
        pad_mask_id = pad_token_id
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_mask_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

use_wandb = False
device = torch.device('cuda:3')

print("Loading train dataset...")

train_dataset = []
nlp = English()
tokenizer = nlp.tokenizer
path = 'data/e2e_data/src1_train.txt'
with open(path, 'r') as ff:
    for row in ff:
        word_lst = row.split('||')[1]
        word_lst = [x.text for x in tokenizer(word_lst)]
        train_dataset.append(word_lst)

train_datasets = Dataset.from_dict({'text': train_dataset})
raw_datasets = train_datasets.train_test_split(0.01)

counter = Counter()
for input_ids in train_dataset:
    counter.update(input_ids)
vocab = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
for k, v in counter.items():
    if v > 10:
        vocab[k] = len(vocab)
raw_datasets.vocab = vocab
tokenizer = vocab
reverse_tokenizer = {v: k for k, v in tokenizer.items()}

raw_datasets['validation'] = raw_datasets['test']
column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

parser = benepar.Parser("benepar_en3")
parser._parser = parser._parser.to(device)
tree_vocab = parser._parser.config["label_vocab"]

diffusion_steps = 200

config = BertConfig()
config.vocab_size = len(tokenizer)
config.input_emb_dim = 16 # TODO
config.tree_vocab_size = len(tree_vocab)
config.train_diff_steps = diffusion_steps

def tokenize_function(examples):
    vocab_dict = raw_datasets.vocab
    sent_lst = []
    for sent in examples['text']:
        input_sentence1 = benepar.InputSentence(words=sent[:63])
        sent_lst.append(input_sentence1)
    parse_lst = list(parser.parse_sents(sent_lst))

    chart_lst = []
    for x in parse_lst:
        chart = chart_from_tree(tree_vocab, x)
        chart_lst.append(chart)
    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
    result_dict = {'input_ids': input_ids, 'chart_lst': chart_lst}
                
    return result_dict

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=column_names
)

def pad_function(group_lst):
    vocab_dict = raw_datasets.vocab
    max_length = 64
    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
    group_lst['parse_chart'] = pad_charts(group_lst['chart_lst'], padding_value=-100)
            
    return group_lst

lm_datasets = tokenized_datasets.map(
    pad_function,
    batched=True,
    num_proc=1,
)

train_dataset = lm_datasets["train"]
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=RandomSampler(train_dataset, generator=torch.Generator()),
    # collate_fn=default_data_collator,
    num_workers=4
)

eval_dataset = lm_datasets["validation"]
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=32,
    sampler=RandomSampler(eval_dataset, generator=torch.Generator()),
    # collate_fn=default_data_collator,
    num_workers=4
)

print("Creating models...")

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
model = TreeControl(config=config, diffusion=diffusion)
model.transformer.embeddings.word_embeddings.load_state_dict(torch.load('data/e2e_data/random_emb.torch'))
model.transformer.embeddings.word_embeddings.weight.requires_grad = False
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())
epochs = 10
if use_wandb:
    wandb.init(project="diffusion-lm")

print("Start training...")
for epoch in range(epochs):
    train_batches = 0
    train_loss = 0
    print('Epoch', epoch + 1, '/', epochs)
    for inputs in tqdm(train_dataloader):
        inputs = inputs.to(device)
        model.train()
        outputs = model(**inputs)
        loss = outputs["loss"]
        train_loss += loss.detach().mean()
        train_batches += 1
        loss.backward()
        optimizer.step()
        model.zero_grad()

        if use_wandb:
            wandb.log({"control/train/loss": loss})

    train_loss = train_loss.cpu()
    if use_wandb:
        wandb.log({"control/train_epoch/loss": train_loss / train_batches})
    print('train epoch loss =', train_loss / train_batches)
