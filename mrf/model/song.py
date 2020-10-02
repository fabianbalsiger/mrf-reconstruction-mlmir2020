"""Implements the method of Song et al., Med Phys, 2019. https://doi.org/10.1002/mp.13727

Unfortunately, no public code of the network is available, so there might be a mismatch between the paper and this implementation.
Indeed, there are several issues in the paper that make the re-implementation difficult.
For instance, Fig. 3 and Sec. 2.C.2 do not correspond in terms of number of convolutional filters in the residual block.
Further, the residual block is not implementable as illustrated in Fig. 3 due to mismatch of number of channels.
Also, there is no mentioning of what kind of activation functions were used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_SONG = 'song'


class Block(nn.Sequential):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.add_module('residual', ResidualBlock(ch_in, ch_out))
        self.add_module('nonlocal', NonLocalBlock(ch_out, ch_out))

    def forward(self, x):
        return super().forward(x)


class ResidualBlock(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=21, stride=1, padding=10)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=21, stride=1, padding=10)

    def forward(self, x):
        # note that this does not correspond entirely with the description and figure. It is not possible
        # to implement it as it is drawn in the figure because the number of channels do not correspond
        # to do the sum of the residual connection.
        x = self.pool(x)
        x = F.relu(self.conv1(x))
        y = F.relu(self.conv2(x))
        y += x
        return y


class NonLocalBlock(nn.Module):
    """Adapted from Wang et al., CVPR, 2018. https://arxiv.org/pdf/1711.07971.pdf"""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()

        self.conv_phi = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv_theta = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv_g = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        phi = self.conv_phi(x)
        theta = self.conv_theta(x)
        theta = theta.permute(0, 2, 1)
        f = F.softmax(torch.matmul(theta, phi), dim=-1)

        g = self.conv_g(x)
        g = g.permute(0, 2, 1)

        y = torch.matmul(f, g)
        y = y.permute(0, 2, 1).contiguous()
        # according to Wang et al., here would be another convolutional layer.
        # but Song et al. do not mention it in the text nor in Fig. 3
        y += x
        return y


class SongModel(nn.Module):

    def __init__(self, ch_in: int = 350, ch_out: int = 5, **kwargs):
        super().__init__()
        self.seq_len = ch_in // 2

        self.conv1 = nn.Conv1d(2, 16, kernel_size=21, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=21, stride=1, padding=0, dilation=1)

        self.block1 = Block(16, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 128)

        self.fc = nn.Linear(128, ch_out)

    def forward(self, x):
        x = torch.stack((x[:, :self.seq_len], x[:, self.seq_len:]), 1)  # two channels (real and imaginary parts)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.avg_pool1d(x, kernel_size=7)  # global average pooling
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x
