import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch.nn import Embedding, Sequential, Linear,Parameter,ReLU

class SimplicialMessagePassingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_edge_conv=False):

        super().__init__()

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

    def forward(self, v0, v1, v2, v3, edge_index0, edge_index1, edge_index2):

        x = self.lin(v0)
        w = self.mlp(v1)
        w= torch.concat([w,w],axis=0)
        out = self.propagate(edge_index0, x=x, w=w)

        out += self.bias
        return self.act(out)


    def propagate(self, edge_index0, x, w):
        edge_index0 = torch.concat([edge_index0,edge_index0.flip([0])],axis=1)
        i, j = edge_index0
        xj = x[j] * w
        out = scatter(xj, i, dim=0)
        return out






class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class cal_total(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(cal_total, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        u = scatter(v, batch, dim=0)
        return u

class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class init_v(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(init_v, self).__init__()
        self.emb = Embedding(100, hidden_channels)
        self.lin1 = Linear(1, 50)
        self.lin2 = Linear(1, hidden_channels)
        self.lin3 = Linear(1, hidden_channels)

    def forward(self, z,z1,z2,z3):
        return self.emb(z),self.lin1(z1.view(-1,1)),self.lin2(z2.view(-1,1)),self.lin3(z3.view(-1,1))


class SMPNN(torch.nn.Module):
    def __init__(self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, out_channels=1,
                 num_filters=128, num_gaussians=50):
        super().__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = init_v(hidden_channels)

        self.update_vs = torch.nn.ModuleList([SimplicialMessagePassingBlock(num_filters, hidden_channels,is_edge_conv=True) for _ in range(num_layers)])

        self.cal_total = cal_total(hidden_channels, out_channels)

        self.reset_parameters()


    def reset_parameters(self):
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.cal_total.reset_parameters()

    def forward(self, batch_data):

        data = batch_data['batch_data']
        adj = batch_data['batch_adj']

        z0, pos0, batch = data.z, data.pos, data.batch
        z1, z1_batch = data.z1, data.z1_batch
        z2, z2_batch = data.z2, data.z2_batch
        z3 = data.z3
        edge_index0, edge_index1, edge_index2 = adj.am0, adj.am1, adj.am2

        if self.energy_and_force:
            pos0.requires_grad_()

        # feature embedding
        v0,v1,v2,v3 = self.init_v(z0,z1,z2,z3)


        for update_v in self.update_vs:
            v0 = update_v(v0, v1, v2, v3, edge_index0, edge_index1, edge_index2)

        # v0 = self.pooling(v0)
        # v1 = self.pooling(v1)
        # v2 = self.pooling(v2)


        u = self.cal_total(v0, batch)
        return u