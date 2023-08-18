import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear,Parameter,ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import GMMConv

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, is_edge_conv=False):
        # is_edge_conv arg controls that if edge features are involved in the message function.
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.is_edge_conv = is_edge_conv
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.act = ReLU()
        self.reset_parameters()

        if is_edge_conv:
            self.mlp = Sequential(  Linear(50, out_channels),
                                    ReLU(),
                                    Linear(out_channels, out_channels))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, e, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if not self.is_edge_conv:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        if self.is_edge_conv:
            w = self.mlp(e)
            out = self.propagate(edge_index, x=x, w=w, norm=None)

        else:
            out = self.propagate(edge_index, x=x, w=None, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias
        return self.act(out)


    def message(self, x_j, w, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        if self.is_edge_conv == True:
            return x_j*w

        else:
            return norm.view(-1, 1) * x_j
