"""Implements the method of Hoppe et al., ISMRM, 2018 (also used for MICCAI 2019 as CNN baseline).

Unfortunately, no public code of the network is available, so there might be a mismatch between the paper and this implementation.
Indeed, the strides of the convolutional filters are not entirely the same due to differences in the MRF sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_HOPPE = 'hoppe'


class ConvolutionLayer(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=0, dilation=1)
        self.batchnorm = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        x = self.batchnorm(x)
        return x


class FullyConnectedLayer(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.fc = nn.Linear(ch_in, ch_out)
        self.batchnorm = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.batchnorm(x)
        return x


class HoppeCNNModel(nn.Module):

    def __init__(self, ch_in: int = 350, ch_out: int = 5, **kwargs):
        super().__init__()

        self.seq_len = ch_in // 2

        # we adapt strides such that the dimensions are (batch_size, 240, 49) after conv4
        # instead of (batch_size, 240, 48) as written in the paper
        self.conv1 = ConvolutionLayer(2, 30, kernel_size=15, stride=1)
        self.conv2 = ConvolutionLayer(30, 60, kernel_size=10, stride=1)
        self.conv3 = ConvolutionLayer(60, 120, kernel_size=5, stride=1)
        self.conv4 = ConvolutionLayer(120, 240, kernel_size=3, stride=3)
        self.fc1 = FullyConnectedLayer(240 * 24, 1000)  # here, we have 24 instead of 23
        self.fc2 = FullyConnectedLayer(1000, 500)
        self.fc3 = FullyConnectedLayer(500, 300)
        self.fc4 = FullyConnectedLayer(300, ch_out)

    def forward(self, x):
        x = torch.stack((x[:, :self.seq_len], x[:, self.seq_len:]), 1)  # two channels (real and imaginary parts)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = F.avg_pool1d(x, kernel_size=2)
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
