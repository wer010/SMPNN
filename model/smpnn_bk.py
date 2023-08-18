import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from math import sqrt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class swish(torch.nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

# for input
class emb(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(emb, self).__init__()
        self.emb = nn.Embedding(100, hidden_channels)
        self.lin1 = nn.Linear(1, hidden_channels)
        self.lin2 = nn.Linear(1, hidden_channels)
        self.lin3 = nn.Linear(1, hidden_channels)

    def forward(self, z,z1,z2,z3):
        return self.emb(z),self.lin1(z1),self.lin2(z2),self.lin3(z3)



class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.act1 = ShiftedSoftplus()
        self.linear2 = nn.Linear(out_features, out_features)
        self.act2 = ShiftedSoftplus()


    def forward(self, x, e,adj_matrix):

        out = torch.matmul(adj_matrix, x)
        out = self.linear1(out)
        out= self.act1(out)
        out = self.linear2(out)
        out = self.act2(out)
        return x+out



class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                 num_spherical, num_radial,
                 num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2

class update_x(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(update_x, self).__init__()
        self.conv1 = GraphConvolutionLayer(in_features, out_features)
        self.conv2 = GraphConvolutionLayer(out_features, out_features)
        self.act = ShiftedSoftplus()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, x, adj_matrix):
        x= self.conv1(x, adj_matrix)
        x = self.act()
        x = self.conv2(x, adj_matrix)
        return x

class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u

class SimplicialMessagePassingBlock(torch.nn.Module):
    def __init__(self, order):
        super(SimplicialMessagePassingBlock, self).__init__()
        self.order = order

    def forward(self, x, adj_matrix):
        for i in self.order:
            x0 = x[0]
        x = self.linear(x)
        x = torch.matmul(adj_matrix, x)
        return x



class SMPNN(torch.nn.Module):

    def __init__(
            self, energy_and_force=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3,
            act=swish, output_init='GlorotOrthogonal', use_node_features=True):
        super(SMPNN, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force


        self.emb = emb(hidden_channels)

        self.smpblocks = torch.nn.ModuleList([SimplicialMessagePassingBlock(order=3) for _ in range(num_layers)])

        # self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for smpblock in self.smpblocks:
            smpblock.reset_parameters()


    def forward(self, batch_data, require_node_features=False):
        self.results=[]

        data = batch_data['batch_data']
        adj = batch_data['batch_adj']

        z, pos, batch = data.z, data.pos, data.batch
        z1, ori1, pos1 = data.z1, data.ori1, data.pos1
        z2, ori2, pos2 = data.z2, data.ori2, data.pos2
        z3, pos3 = data.z3, data.pos3
        am0, am1, am2 = adj.am0, adj.am1, adj.am2

        if self.energy_and_force:
            pos.requires_grad_()
        # edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes = z.size(0)
        # dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)

        x,x1,x2,x3 = self.emb(z,z1,z2,z3)

        node_feature = {'s0':x,
                        's1':x1,
                        's2':x2,
                        's3':x3,}
        edge_feature = {'pos0':pos,
                        'pos1':pos1,
                        'pos2':pos2,
                        'pos3':pos3,}
        # after embedding, then simplicial message passing
        for blk in zip(self.smpblocks):

            e0,e1,e2,e3 = blk(node_feature)
            x,x1,x2,x3 = update_x(x,x1,x2,x3, e1,e2,e3, am0, am1, am2)
            self.results.append(x)

        u = self.update_u(x, batch)

        if require_node_features:
            return x[0]
        else:
            return u
