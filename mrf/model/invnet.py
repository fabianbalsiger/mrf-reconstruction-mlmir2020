import FrEIA.framework as freia_fw
import FrEIA.modules as freia_mod
import torch.nn as nn


def get_nodes(ndim, nb_blocks=4, hidden_layer=128, small_block=False, permute=True):
    nodes = [freia_fw.InputNode(ndim, name='input')]

    for i in range(nb_blocks):
        def F_fully_connected_wrapper(ch_in, ch_out):
            if not small_block:
                return freia_mod.F_fully_connected(ch_in, ch_out, internal_size=hidden_layer)
            return F_fully_connected_small(ch_in, ch_out, internal_size=hidden_layer)

        nodes.append(freia_fw.Node([nodes[-1].out0],
                                   freia_mod.RNVPCouplingBlock,
                                   {'subnet_constructor': F_fully_connected_wrapper,
                                    },
                                   name='coupling_{}'.format(i)))

        if permute:
            nodes.append(
                freia_fw.Node([nodes[-1].out0],
                              freia_mod.PermuteRandom,
                              {'seed': i},
                              name='permute_{}'.format(i)))

    nodes.append(freia_fw.OutputNode([nodes[-1].out0],
                                     name='output'))
    return nodes


class F_fully_connected_small(nn.Module):

    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        super(F_fully_connected_small, self).__init__()
        if not internal_size:
            internal_size = 2*size

        self.d1 = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

        self.nl1 = nn.ReLU()

    def forward(self, x):
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.fc3(out)
        return out


def get_invnet(ndim, nb_blocks=4, hidden_layer=128, small_block=False, permute=True, verbose=False):
    return freia_fw.ReversibleGraphNet(get_nodes(ndim, nb_blocks, hidden_layer, small_block, permute), verbose=verbose)
