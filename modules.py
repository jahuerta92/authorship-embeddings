import torch
import pytorch_lightning as pl

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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