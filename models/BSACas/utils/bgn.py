from os import WSTOPPED
import numpy as np
from numpy.core.fromnumeric import reshape
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import batch
from torch_geometric.nn import global_add_pool, GCNConv, GATv2Conv
from torch_scatter import scatter_add
from scipy import sparse as sp
import numpy as np


class GPNConv(torch.nn.Module):
    def __init__(self, nn, input_size=None, hidden_size=None, gates_num=4):
        super(GPNConv, self).__init__()
        # self.nn = NNLinear(input_size, hidden_size, gates_num)
        if nn is not None:
            self.nn = nn
        else:
            self.nn = torch.nn.Linear(input_size, hidden_size * gates_num, bias=True)

    def forward(self, x, edge_index, adj=None):
        # x = x.unsqueeze(-1) if x.dim()==1 else x
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = x + out
        out = self.nn(out)
        return out


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            print('single linear')
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class Identical(nn.Module):
    def __init__(self):
        '''a layer doing nothing'''
        super(Identical, self).__init__()

    def forward(self, x):
        return x


class BGA(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, max_len=None, return_middle_feature=False, gcn_type=None, usr_attn_type=None, batch_step_dim=None, n_vocabulary=None):
        super(BGA, self).__init__()

        self.batch = batch_step_dim[0]
        self.n_step = batch_step_dim[1]
        self.usr_attn_type = usr_attn_type
        self.gpn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.max_cascade_len = max_len
        self.hidden_dim = hidden_dim
        self.num_heads = 2
        # graph attention layer
        assert gcn_type is not None
        for layer in range(num_layers - 1):
            if gcn_type == 'self':
                if layer == 0:
                    mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
                else:
                    mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

                gpn_conv = GPNConv(mlp, 0)
                self.gpn_layers.append(gpn_conv)

                batch_norm = nn.BatchNorm1d(hidden_dim)
                self.batch_norms.append(batch_norm)
            elif gcn_type == 'geo':
                print('geometric gcn')
                if layer == 0:
                    gpn_conv = GCNConv(input_dim, hidden_dim)
                else:
                    gpn_conv = GCNConv(hidden_dim, hidden_dim)
                self.gpn_layers.append(gpn_conv)
                self.batch_norms.append(Identical())

        # attention layers
        if usr_attn_type == 'GAT':
            # self.gcn_conv = GCNConv(hidden_dim,hidden_dim)
            self.atten_self = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False, heads=self.num_heads, dropout=0.01, concat=True)
            self.atten = GATv2Conv(hidden_dim*self.num_heads, hidden_dim, add_self_loops=False, heads=self.num_heads, dropout=0.001, concat=False)

        else:
            self.attn = None

        self.return_middle_feature = return_middle_feature

        self.linears_prediction = torch.nn.ModuleList()
        middle_dim = 32
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, middle_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, middle_dim))

        self.output_layer = nn.Linear(middle_dim, output_dim)

    def forward(self, batch_x):
        x = batch_x.x
        A = batch_x.edge_index
        batch_index = batch_x.batch
        atten_edge_index = batch_x.atten_edge_index
        del batch_x
        w = None
        h = x
        hidden_rep = [h]
        # GCN operation
        for layer in range(self.num_layers - 1):
            h = self.gpn_layers[layer](h, A)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        output_h = 0
        weights = ['edge_index', 'value']
        atten_out = None
        if self.usr_attn_type == 'GAT':
            output_emb = hidden_rep[-1]

            # print(output_emb.size()[0])
            # <*,features>
            # output_emb=self.gcn_conv(output_emb,A)
            output_emb = self.atten_self(output_emb, A)
            output_emb = F.relu(output_emb)
            atten_out, w = self.atten(output_emb, atten_edge_index, return_attention_weights=True)
            weights[0], weights[1] = w[0].cpu(), w[1].cpu()
            atten_out = F.relu(atten_out)
            hidden_rep.append(atten_out)
        for layer, h in enumerate(hidden_rep):
            # get x and h do residual
            if not (layer == 0 or layer == len(hidden_rep) - 1):
                continue
            # put pool layer here
            h_pool = global_add_pool(h, batch=batch_index)
            # h_pool = torch.sum(h, dim=0, keepdim=True)
            output_h += self.linears_prediction[layer if layer == 0 else -1](h_pool)

        output_h = F.relu(output_h)
        outputs = self.output_layer(output_h)

        if self.return_middle_feature:
            return outputs, output_h

        return outputs, atten_out, (A.tolist(), weights)


# if __name__ == '__main__':
#     from torch_geometric.datasets import TUDataset
#     from torch_geometric.data import DataLoader

#     dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     data = None
#     for d in loader:
#         data = d
#         break
#     gpn = GPN(2, 2, 3, 8, 5)
#     outputs = gpn(data.x, data.edge_index, data.batch)
#     print(outputs.shape)


# GRU->transformer
# NodeAttention-> remove RNN+CNN
# Temporal(RNNCNN) -> remove NodeAttention
# Temporal(CNN) -> remove RNN
# Temporal(RNN) -> remove CNN
# Structural -> remove RNN & nodeattention
# * -> loss_ + diff_loss_
