import torch
import numpy as np

from random import shuffle
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
from tqdm import tqdm

def join_text(list_text):
    return ' '.join(list_text)

class ContrastDataset(Dataset):
    def __init__(self, text_data, steps, window=512):
        self.text_data = text_data
        self.steps = steps
        self.window = window
    
        self.orig_authors = text_data.id.unique().tolist()
        self.n_authors = len(self.orig_authors)
        self.text_data = self.text_data.set_index(['id', 'unique_id'])
        self.authors = self.populate_authors()
    
    def populate_authors(self):
        n = self.steps//self.window
        if self.steps%self.window != 0:
            n += 1
        expanded = self.orig_authors * n

        return expanded[:self.steps]

    def __len__(self):
        return len(self.authors)

class BalancedContrastDataset(ContrastDataset):
    def generate_dataset_balance(self):
        keys = []
        for author in self.orig_authors:
            k, _ = author.split('_')
            if k not in keys:
                keys.append(k)
        
        count_dict = {k: 0 for k in keys}
        for author in self.orig_authors:
            k, _ = author.split('_')
            count_dict[k] += 1

        balanced_authors = []
        for author in self.orig_authors:
            k, _ = author.split('_')
            balanced_authors.append(1/(len(keys)*count_dict[k]))        

        return balanced_authors

    def populate_authors(self):
        balanced_orig_authors = self.generate_dataset_balance()
        n = self.steps//self.window
        if self.steps%self.window != 0:
            n += 1
        self.authors = []
        for _ in range(n):
            next_authors = np.random.choice(self.orig_authors, self.window, p=balanced_orig_authors).tolist()
            self.authors += next_authors
            
        return self.authors[:self.steps]

class TextContrastDataset(ContrastDataset):
    def __getitem__(self, i):
        auth = self.authors[i]
        anchor, replica = self.text_data.loc[auth].sample(2).decoded_text.tolist()
        
        return anchor, replica

class PretokenizedContrastDataset(ContrastDataset):
    def __getitem__(self, i):
        auth = self.authors[i]
        anchor, replica = self.text_data.loc[auth].sample(2).pretokenized_text.tolist()

        return literal_eval(anchor), literal_eval(replica)

class TextCollator:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __call__(self, texts):
        anchors, replicas = list(zip(*texts))
        config = dict(padding='max_length',
                      return_tensors='pt',
                      truncation=True,
                      max_length=self.max_len,)

        encoded_anchors = self.tokenizer(list(anchors), **config)
        encoded_replicas = self.tokenizer(list(replicas), **config)

        return (encoded_anchors.input_ids,
                encoded_anchors.attention_mask,
                encoded_replicas.input_ids,
                encoded_replicas.attention_mask,
                )
    
class PretokenizedCollator:
    def __init__(self, max_len=512):
        self.max_len = max_len

    def pad_list(self, list, pad_value=0):
        return torch.Tensor(list + [pad_value] * (self.max_len - len(list)))

    def __call__(self, pretokenized_texts):
        anchors, replicas = list(zip(*pretokenized_texts))
        
        anchor_ids = torch.stack([self.pad_list(anchor) for anchor in anchors], dim=0).int()
        replica_ids = torch.stack([self.pad_list(replica) for replica in replicas], dim=0).int()
        anchor_mask = torch.ones_like(anchor_ids).int()
        replica_mask = torch.ones_like(replica_ids).int()

        return (anchor_ids,
                anchor_mask,
                replica_ids, 
                replica_mask)

def build_dataset(dataframe,
                  steps,
                  tokenizer=None,
                  max_len=128,
                  batch_size=16, 
                  num_workers=4, 
                  prefetch_factor=4,
                  shuffle=False, 
                  mode='pretokenized'):
    if mode == 'pretokenized' or tokenizer is None:
        collator = PretokenizedCollator(max_len=max_len)
        dataset = PretokenizedContrastDataset(dataframe, steps, window=batch_size)
    elif mode == 'text' or tokenizer is not None:
        collator = TextCollator(tokenizer, max_len=max_len)
        dataset = TextContrastDataset(dataframe, steps, window=batch_size)
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      collate_fn=collator)
                     