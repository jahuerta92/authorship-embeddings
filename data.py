import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from ast import literal_eval

def join_text(list_text):
    return ' '.join(list_text)

class ContrastDataset(Dataset):
    def __init__(self, text_data, steps):
        self.text_data = text_data
        self.authors = text_data.id.unique().tolist()
        self.text_data = self.text_data.set_index(['id', 'unique_id'])
        self.steps = steps

    def __len__(self):
        return self.steps

class TextContrastDataset(ContrastDataset):
    def __getitem__(self, i):
        n_auth = len(self.authors)
        auth = self.authors[i%n_auth]
        anchor, replica = self.text_data.loc[auth].sample(2).decoded_text.tolist()
        
        return anchor, replica

class PretokenizedContrastDataset(ContrastDataset):
    def __getitem__(self, i):
        n_auth = len(self.authors)
        auth = self.authors[i%n_auth]
        anchor, replica = self.text_data.loc[auth].sample(2).pretokenized_text.tolist()

        return anchor, replica

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
                  shuffle=False, 
                  batch_size=16, 
                  num_workers=4, 
                  prefetch_factor=4, 
                  mode='pretokenized'):
    if mode == 'pretokenized' or tokenizer is None:
        collator = PretokenizedCollator(max_len=max_len)
        dataset = PretokenizedContrastDataset(dataframe, steps)
    elif mode == 'text' or tokenizer is not None:
        collator = TextCollator(tokenizer, max_len=max_len)
        dataset = TextContrastDataset(dataframe, steps)
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      collate_fn=collator)
                     