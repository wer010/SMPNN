import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from model.gcn import GCNConv
from torch_geometric.nn import SchNet

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



class MPNN(torch.nn.Module):
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

        self.init_v = Embedding(100, hidden_channels)
        # self.dist_emb = emb(0.0, cutoff, num_gaussians)
        self.dist_emb = Linear(1, num_gaussians)


        self.update_vs = torch.nn.ModuleList([GCNConv(num_filters, hidden_channels,is_edge_conv=True) for _ in range(num_layers)])

        self.cal_total = cal_total(hidden_channels, out_channels)

        self.reset_parameters()


    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.cal_total.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)

        # feature embedding
        e = self.dist_emb(dist.view(-1,1))
        v = self.init_v(z)

        for update_v in self.update_vs:
            v = update_v(v, e, edge_index)

        u = self.cal_total(v, batch)
        return u