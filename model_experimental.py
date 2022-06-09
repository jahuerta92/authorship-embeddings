import torch
import math
import copy

from transformers import get_linear_schedule_with_warmup, Adafactor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from modules import DynamicLSTM
from losses import infonce_loss, flatnce_loss, oneway_infonce_loss

def switch_gradient(model, freeze: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(freeze)

class ContrastivePretrain(pl.LightningModule):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=True,
                 minibatch_size=128,
                 label_smoothing=0.25,
                 ):
        super().__init__()

        # Save hyperparameters for training

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler
        self.minibatch_size = minibatch_size
        self.label_smoothing = label_smoothing
        self.loss_func = infonce_loss

        self.save_hyperparameters()

        self.transformer = transformer

        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay,
                          )

        if self.enable_scheduler:
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.num_warmup_steps,
                                                        num_training_steps=self.num_training_steps,
                                                        )
                                                        
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": 'linear_schedule_with_warmup',
            }
            return {'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler_config,
                    }
        else:
            return {'optimizer': optimizer,
                    #'lr_scheduler': lr_scheduler_config,
                    }
    
    def training_step(self, train_batch, batch_idx):
        anchors_input_ids, anchors_attention_mask, replicas_input_ids, replicas_attention_mask = train_batch
        
        optimizer = self.optimizers()
        
        loss_tracker, acc_tracker = [], []

        global_batch_size = len(anchors_input_ids)
        n = int(math.ceil( global_batch_size/ self.minibatch_size))

        mb_anchors_input_ids = torch.chunk(anchors_input_ids, n)
        mb_anchors_attention_mask = torch.chunk(anchors_attention_mask, n)
        mb_replicas_input_ids = torch.chunk(replicas_input_ids, n)
        mb_replicas_attention_mask = torch.chunk(replicas_attention_mask, n)

        with torch.no_grad():
            anchors = torch.cat([self(id, msk) for id, msk in zip(mb_anchors_input_ids, mb_anchors_attention_mask)], dim=0)
            replicas = torch.cat([self(id, msk) for id, msk in zip(mb_replicas_input_ids, mb_replicas_attention_mask)], dim=0)

        optimizer.zero_grad()

        for j, (a_ids, a_msk, r_ids, r_msk) in enumerate(zip(mb_anchors_input_ids, mb_anchors_attention_mask, mb_replicas_input_ids, mb_replicas_attention_mask)):
            anchors_rep = copy.deepcopy(anchors)
            replicas_rep = copy.deepcopy(replicas)
            anchors_rep[(j * self.minibatch_size):((j+1) * self.minibatch_size)] = self(a_ids, a_msk)
            replicas_rep[(j * self.minibatch_size):((j+1) * self.minibatch_size)] = self(r_ids, r_msk)

            loss, acc = self.loss_func(anchors_rep, replicas_rep, self.temperature, self.label_smoothing)
            loss_tracker.append(loss)
            acc_tracker.append(acc)
            self.manual_backward(loss)

        with torch.no_grad():
            self.log(f'train/infonce_loss', sum(loss_tracker) / len(loss_tracker))
            self.log(f'train/infonce_acc', sum(acc_tracker) / len(acc_tracker))
        
        optimizer.step()

        if self.enable_scheduler:
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()

        return loss

    def validation_step(self, val_batch, batch_idx):

        anchors_input_ids, anchors_attention_mask, replicas_input_ids, replicas_attention_mask = val_batch
        
        n = int(math.ceil(len(anchors_input_ids) / self.minibatch_size*4))

        mb_anchors_input_ids = torch.chunk(anchors_input_ids, n)
        mb_anchors_attention_mask = torch.chunk(anchors_attention_mask, n)
        mb_replicas_input_ids = torch.chunk(replicas_input_ids, n)
        mb_replicas_attention_mask = torch.chunk(replicas_attention_mask, n)

        anchors = torch.cat([self(id_, msk) for id_, msk in zip(mb_anchors_input_ids, mb_anchors_attention_mask)], dim=0)
        replicas = torch.cat([self(id_, msk) for id_, msk in zip(mb_replicas_input_ids, mb_replicas_attention_mask)], dim=0)
        
        loss, acc = self.loss_func(anchors, replicas, self.temperature, self.label_smoothing)

        self.log(f'valid/infonce_loss', loss)
        self.log(f'valid/infonce_acc', acc)

        return loss

    def predict_step(self, pred_batch, batch_idx):
        anchors, _, _ = pred_batch

        return self(anchors.input_ids, anchors.attention_mask)

class ContrastiveTransformer(ContrastivePretrain):
    def forward(self, input_ids, attention_mask=None):
        return  self.transformer(input_ids, attention_mask=attention_mask).pooler_output

class ContrastiveLSTMTransformer(ContrastivePretrain):
    def __init__(self, transformer,
                 learning_rate=3e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=False,
                 minibatch_size=256,
                 ):
        super().__init__(transformer,
                         learning_rate,
                         weight_decay,
                         num_warmup_steps,
                         num_training_steps,
                         enable_scheduler,
                         minibatch_size,
                         )
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        if 'T5Config' in str(transformer.config_class):
            orig_size = transformer.config.d_model
            embed_size = orig_size//4
        else:
            orig_size = transformer.config.hidden_size
            embed_size = orig_size//2

        self.pooler = DynamicLSTM(orig_size,
                                  embed_size,
                                  dropout=.1,
                                  bidirectional=True)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.transformer(input_ids, attention_mask= attention_mask).last_hidden_state

        return self.pooler(embeds, attention_mask)

class ContrastiveDenseTransformer(ContrastivePretrain):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=False,
                 minibatch_size=256,
                 **kwargs,
                 ):
        super().__init__(transformer,
                         learning_rate,
                         weight_decay,
                         num_warmup_steps,
                         num_training_steps,
                         enable_scheduler,
                         minibatch_size,
                         )
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        if 'T5Config' in str(transformer.config_class):
            orig_size = transformer.config.d_model
            embed_size = orig_size
        else:
            orig_size = transformer.config.hidden_size
            embed_size = orig_size

        self.pooler = torch.nn.Linear(orig_size, embed_size)

class ContrastiveMeanDenseTransformer(ContrastiveDenseTransformer):
    def forward(self, input_ids, attention_mask=None):
        embeds = self.transformer(input_ids, attention_mask).last_hidden_state

        if attention_mask is None:
            embed = embeds.mean(1)
        else:
            embed = (embeds*attention_mask.unsqueeze(-1)).sum(1) / \
                attention_mask.sum(1).unsqueeze(-1)

        return self.pooler(embed)

class ContrastiveMaxDenseTransformer(ContrastiveDenseTransformer):
    def forward(self, input_ids, attention_mask=None):
        embeds = self.transformer(input_ids, attention_mask).last_hidden_state

        embed = embeds.max(1)[0]

        return self.pooler(embed)