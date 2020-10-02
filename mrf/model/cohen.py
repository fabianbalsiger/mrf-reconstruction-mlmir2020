"""Implements the method of Cohen et al., Magn Reson Med, 2018. https://doi.org/10.1002/mrm.27198

Unfortunately, no public code of the network is available, so there might be a mismatch between the paper and this implementation.
We choose to use ReLUs instead of tanh activation functions because it performed slightly better.
Also, we do not use a sigmoid at the output layer.
"""
import torch.nn as nn
import torch.nn.functional as F


MODEL_COHEN = 'cohen'


class CohenModel(nn.Module):

    def __init__(self, ch_in: int = 350, ch_out: int = 5, **kwargs):
        super().__init__()
        hidden_size = kwargs.get('hidden_size', 300)

        self.fc1 = nn.Linear(ch_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, ch_out)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x
