from os import WSTOPPED
import numpy as np
from numpy.core.fromnumeric import reshape
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import batch
from torch_geometric.nn import global_add_pool, GCNConv
from torch_scatter import scatter_add
from scipy import sparse as sp
import numpy as np
from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
"""_summary_
This is the implementation of bidirectional graph neural network
The codes is based on torch_geometric and much referred its codes. Thanks for Thomas Kipf.
Please cite his paper Fast Graph Representation Learning with PyTorch Geometric
and our paper BSACas.
"""

class ATTConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(ATTConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                            weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights: bool = None):
    
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GANConv(torch.nn.Module):
    def __init__(self, nn, input_size=None, hidden_size=None, gates_num=4):
        super(GANConv, self).__init__()
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
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, max_len=None, return_middle_feature=False, gcn_type=None, usr_attn_type=None, batch_step_dim=None):
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

                gpn_conv = GANConv(mlp, 0)
                self.gpn_layers.append(gpn_conv)

                batch_norm = nn.BatchNorm1d(hidden_dim)
                self.batch_norms.append(batch_norm)

        # attention layers
        if usr_attn_type == 'ATT':
            self.atten_self = ATTConv(hidden_dim, hidden_dim, add_self_loops=False, heads=self.num_heads, dropout=0.01, concat=True)
            self.atten = ATTConv(hidden_dim*self.num_heads, hidden_dim, add_self_loops=False, heads=self.num_heads, dropout=0.001, concat=False)
        elif usr_attn_type == 'GCN':
            self.gcn_conv = GCNConv(hidden_dim,hidden_dim)
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
        if self.usr_attn_type == 'ATT':
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
