import os
import json
import torch
import numpy as np
from spacy.lang.en import English
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


class TextDataset(Dataset):
    def __init__(self, text_datasets, resolution, data_args, eigen_transform=None,
                 mapping_func=None, model_emb=None, noise_level=0):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        self.noise_level = noise_level

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr = np.array(self.text_datasets['train'][idx]['hidden_states'], dtype=np.float32)

        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)

        if self.noise_level > 0:
            arr = arr + self.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

        out_dict = {}
        out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])

        return arr, out_dict


def helper_tokenize_encode(sentence_lst, vocab_dict, model, seqlen):
    result_train_lst = []
    group_lst = defaultdict(list)

    with torch.no_grad():
        for input_ids in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)

        concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}
        total_length = len(concatenated_examples[list(group_lst.keys())[0]])
        block_size = seqlen
        total_length = (total_length // block_size) * block_size
            
        group_lst = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        for input_ids in group_lst['word_ids']:
            hidden_state = model(torch.tensor(input_ids))
            result_train_lst.append({'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist()})

    return result_train_lst


def get_corpus(path, in_channel, model, image_size, split='train', load_vocab=None):
    sentence_lst = []
    nlp = English()
    tokenizer = nlp.tokenizer
    path = f'{path}/src1_{split}.txt'
          
    with open(path, 'r') as ff:
        for row in ff:
            word_lst = row.split('||')[1]
            word_lst = [x.text for x in tokenizer(word_lst)]
            sentence_lst.append(word_lst)

    if load_vocab is None:
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)

    path_save_vocab = f'{path}/vocab.json'
    if load_vocab is None:
        vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
        for k, v in counter.items():
            if v > 10:
                vocab_dict[k] = len(vocab_dict)

        with open(path_save_vocab, 'w') as f:
            json.dump(vocab_dict, f)
    else:
        vocab_dict = load_vocab
        if not os.path.exists(path_save_vocab):
            if isinstance(vocab_dict, dict):
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                assert vocab_dict['START'] == 0
            elif isinstance(vocab_dict, PreTrainedTokenizerFast):
                vocab_dict.save_pretrained(path)

    path_save = f'{path}/random_emb.torch'
    if model is None:
        model = torch.nn.Embedding(len(vocab_dict), in_channel)
        torch.nn.init.normal_(model.weight)
        torch.save(model.state_dict(), path_save)

    if not os.path.exists(path_save):
        torch.save(model.state_dict(), path_save)

    result_train_lst = helper_tokenize_encode(sentence_lst, vocab_dict, model, image_size**2)
    
    return {'train': result_train_lst}


def load_data_text(batch_size, image_size, model, split, load_vocab):
    training_data, model = get_corpus(
        'data/e2e_data', 16, model, image_size, 
        split=split, load_vocab=load_vocab
    )
    data_loader = DataLoader(
        TextDataset(training_data, image_size),
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )
    while True:
        yield from data_loader
