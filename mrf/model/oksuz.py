"""Implements the method of Oksuz et al., ISBI, 2019. https://doi.org/10.1109/ISBI.2019.8759502

Unfortunately, no public code of the network is available, so there might be a mismatch between the paper and this implementation.
"""
import torch
import torch.nn as nn


MODEL_OKSUZ = 'oksuz'


class OksuzModel(nn.Module):

    def __init__(self, ch_in: int = 350, ch_out: int = 5, **kwargs):
        super().__init__()
        self.seq_len = ch_in // 2
        hidden_size = kwargs.get('hidden_size', 100)

        input_size = 2  # real and imaginary parts
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                          batch_first=True)
        self.fc = nn.Linear(self.seq_len * hidden_size, ch_out)

    def forward(self, x):
        x = torch.stack((x[:, :self.seq_len], x[:, self.seq_len:]), -1)  # two features real and imaginary parts)
        x, _ = self.gru(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
