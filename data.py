import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

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

    def __getitem__(self, i):
        n_auth = len(self.authors)
        auth = self.authors[i%n_auth]
        anchor, replica = self.text_data.loc[auth].sample(2).text.tolist()
        
        return anchor, replica
    
class TextCollator:
    def __init__(self, tokenizer, max_len=196):
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
    
    
def build_dataset(dataframe, tokenizer, steps, max_len=128, shuffle=False, batch_size=16, num_workers=4, prefetch_factor=4, samples_per_line=1):
    data = ContrastDataset(dataframe, steps)
    
    return DataLoader(data,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      collate_fn=TextCollator(tokenizer, max_len=max_len),
                     )