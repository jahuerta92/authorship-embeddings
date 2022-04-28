import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl
import torch.nn.functional as F

class DynamicLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size, self.hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
    def forward(self, x, attention_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:-1], device=self.device)

        seq_lens = attention_mask.sum(-1)
        batch_size = attention_mask.shape[0]
        seq_len = attention_mask.shape[1]

        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort.cpu(), batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        batch_indices = torch.arange(0, batch_size, device=self.device)
        seq_indices = seq_lens - 1

        y_split = y.view(batch_size, seq_len, 2, self.hidden_size)

        output = torch.cat(
            [y_split[batch_indices, seq_indices, 0], y_split[batch_indices, 0, 1]], dim=-1)

        return output

class ContrastivePretrain(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
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

    def infonce_loss(self, a, b):
        batch_size = a.shape[0]
        logits = (a @ b.T) * torch.exp(self.temperature).clamp(max=100)
        labels = torch.arange(0, batch_size, device=self.device)

        loss = (F.cross_entropy(logits.T, labels).mean() +
                F.cross_entropy(logits, labels).mean()) / 2
        
        with torch.no_grad():
            preds = F.softmax(logits, dim=1).argmax(-1)
            preds_t = F.softmax(logits.T, dim=1).argmax(-1)

            accuracy = (torch.sum(preds == labels) +
                        torch.sum(preds_t == labels)) / (batch_size * 2)

        return loss, accuracy

    def eval_batch(self, batch, mode='train'):
        anchors_input_ids, anchors_attention_mask, replicas_input_ids, replicas_attention_mask = batch

        anchor_embeds = self(anchors_input_ids, anchors_attention_mask)
        replicas_embeds = self(replicas_input_ids, replicas_attention_mask)

        loss, acc = self.infonce_loss(anchor_embeds, replicas_embeds)

        self.log(f'{mode}/infonce_loss', loss)
        self.log(f'{mode}/infonce_acc', acc)

        return loss

    def training_step(self, train_batch, batch_idx):
        return self.eval_batch(train_batch)

    def validation_step(self, val_batch, batch_idx):
        return self.eval_batch(val_batch, mode='valid')

    def predict_step(self, pred_batch, batch_idx):
        anchors, _, _ = pred_batch

        return self(anchors.input_ids, anchors.attention_mask)
        

class ContrastiveLSTMHead(ContrastivePretrain):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.save_hyperparameters()

        self.transformer = transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

        embed_size = transformer.config.hidden_size//2
        self.pooler = DynamicLSTM(transformer.config.hidden_size,
                                  embed_size,
                                  dropout=.1,
                                  bidirectional=True)
        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeds = self.transformer(
                input_ids, attention_mask).last_hidden_state

        return F.normalize(self.pooler(embeds, attention_mask))

class ContrastiveDenseHead(ContrastivePretrain):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.save_hyperparameters()

        self.transformer = transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

        embed_size = transformer.config.hidden_size
        self.pooler = torch.nn.Linear(embed_size, embed_size)
        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))

class ContrastiveMeanDenseHead(ContrastiveDenseHead):
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeds = self.transformer(
                input_ids, attention_mask).last_hidden_state

            if attention_mask is None:
                embed = embeds.mean(1)
            else:
                embed = (embeds*attention_mask.unqueeze(-1)).sum(1) / \
                    attention_mask.sum(1).unsqueeze(-1)

        return F.normalize(self.pooler(embed))

class ContrastiveMaxDenseHead(ContrastiveDenseHead):
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeds = self.transformer(
                input_ids, attention_mask).last_hidden_state

            embed = embeds.max(1)

        return F.normalize(self.pooler(embed))

'''
class ContrastivePretrainDense(pl.LightningModule):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.save_hyperparameters()

        self.transformer = transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

        embed_size = transformer.config.hidden_size
        self.pooler = torch.nn.Linear(embed_size, embed_size)
        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))

        self.switch_finetune(False)
        # self.norm = torch.nn.LayerNorm(embed_size)

    def switch_finetune(self, switch=True):
        for param in self.transformer.parameters():
            param.requires_grad = switch

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeds = self.transformer(
                input_ids, attention_mask).last_hidden_state

            if attention_mask is None:
                embed = embeds.mean(1)
            else:
                embed = (embeds*attention_mask.unqueeze(-1)).sum(1) / \
                    attention_mask.sum(1).unsqueeze(-1)

        return F.normalize(self.pooler(embed))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
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

    def infonce_loss(self, a, b):
        batch_size = a.shape[0]
        logits = (a @ b.T) * torch.exp(self.temperature).clamp(max=100)
        labels = torch.arange(0, batch_size, device=self.device)

        loss = (F.cross_entropy(logits.T, labels).mean() +
                F.cross_entropy(logits, labels).mean()) / 2

        with torch.no_grad():
            preds = F.softmax(logits, dim=1).argmax(-1)
            preds_t = F.softmax(logits.T, dim=1).argmax(-1)

            accuracy = (torch.sum(preds == labels) +
                        torch.sum(preds_t == labels)) / (batch_size * 2)

        return loss, accuracy

    def eval_batch(self, batch, mode='train'):
        anchors_input_ids, anchors_attention_mask, replicas_input_ids, replicas_attention_mask = batch

        anchor_embeds = self(anchors_input_ids, anchors_attention_mask)
        replicas_embeds = self(replicas_input_ids, replicas_attention_mask)

        loss, acc = self.infonce_loss(anchor_embeds, replicas_embeds)

        self.log(f'{mode}/infonce_loss', loss)
        self.log(f'{mode}/infonce_acc', acc)

        return loss

    def training_step(self, train_batch, batch_idx):
        return self.eval_batch(train_batch)

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            return self.eval_batch(val_batch, mode='valid')

    def predict_step(self, pred_batch, batch_idx):
        anchors, _, _ = pred_batch

        return self(anchors.input_ids, anchors.attention_mask)
        '''
