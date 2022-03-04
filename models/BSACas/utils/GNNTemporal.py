import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
from .bgn import BGA, MLP
from torch.autograd import Variable
import math
"""_summary_
This is the implementation of bidirectional graph neural network with temporal attention.
Please cite our paper BSACas.

"""

class BSA(nn.Module):
    def __init__(self,
                 n_seq,
                 input_features,
                 max_len,
                 batch_size=32,
                 emb_size=128,
                 hidden_size=128,
                 num_layers=2,
                 gnn_out_features=128,
                 gnn_mlp_hidden=32,
                 bidirected=True,  # disable when using transformer
                 atten_type=None,  # naive-self-attn or GAT
                 device=torch.device('cpu'),
                 rnn='gru'):
        super(BSA, self).__init__()
        self.n_seq = n_seq
        self.rnn = rnn
        num_layers_rnn = 1
        dropout = 0.1
        mlp_hidden = gnn_mlp_hidden
        self.gnn_out_features = gnn_out_features
        num_mlp_layers = 2
        num_layers = 2
        self.gnn = BGA(num_layers,
                       num_mlp_layers,
                       input_dim=input_features,
                       hidden_dim=mlp_hidden,
                       output_dim=self.gnn_out_features,
                       return_middle_feature=False,
                       max_len=max_len,
                       gcn_type='self',
                       usr_attn_type=atten_type,  # node attentions
                       batch_step_dim=[batch_size, n_seq])

        self.diff_mlp = MLP(2, self.gnn_out_features, 64, 1)

        self.feature_linear = nn.Linear(gnn_out_features, emb_size)

        num_direction = 1
        if bidirected and self.rnn != 'transformer':
            num_direction = 2
        self.num_direction = num_direction
        if self.rnn == 'gru':
            self.rnn_model = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers_rnn, bidirectional=bidirected, batch_first=True)
        elif self.rnn == 'LSTM':
            self.rnn_model = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers_rnn, bidirectional=bidirected, batch_first=True)
        elif self.rnn == 'transformer':
            print('emb_size is not avalible when using transformer')
            d_model = emb_size
            emb_size = hidden_size
            nhead = 2
            self.emb_position = nn.Embedding(num_embeddings=n_seq, embedding_dim=emb_size)
            self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dropout=dropout, dim_feedforward=hidden_size)  # , activation='gelu')
            self.encoder_norm = torch.nn.LayerNorm(emb_size)
            self.rnn_model = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers, norm=self.encoder_norm)
            # self.position_emcoding = PositionalEncoding(emb_size, 0.001)
        else:
            self.rnn_model = None
            bidirected = False

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        conv_out_dim = (hidden_size*num_direction) if self.rnn_model else emb_size
        Ks = [7, 5]if self.rnn_model else [2]
        out_c = 64
        self.convs = []
        for K in Ks:
            self.convs.append([nn.Conv1d(conv_out_dim, out_c, K, padding=K//2).to(device), nn.Conv1d(out_c, conv_out_dim, 3, padding=3//2).to(device)])

        if self.rnn == 'gru':
            self.hidden = torch.zeros([self.num_layers * self.num_direction, self.batch_size, self.hidden_size], device=self.device)
        else:
            self.hidden = (torch.zeros([self.num_layers * self.num_direction, self.batch_size, self.hidden_size],
                                       device=self.device), torch.zeros([self.num_layers * self.num_direction, self.batch_size, self.hidden_size], device=self.device))

        self.linear = nn.Linear(conv_out_dim*len(Ks), conv_out_dim*len(Ks)//2)  # .to(torch.device('cuda'))
        self.linear2 = nn.Linear(conv_out_dim*len(Ks)//2, 1)

    def init_hidden(self):
        if self.rnn == 'gru':
            return torch.zeros(size=[self.num_layers * self.num_direction, self.batch_size, self.hidden_size], device=self.device)
        else:
            return (torch.zeros(size=[self.num_layers * self.num_direction, self.batch_size, self.hidden_size],
                                device=self.device), torch.zeros(size=[self.num_layers * self.num_direction, self.batch_size, self.hidden_size], device=self.device))

    def detach_hidden(self, h):
        """detach a hidden in case backward for the second 	time"""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.detach_hidden(v) for v in h)

    def forward(self, batch_x, position_ids=None, return_node_F=False):
        # x(batch_graph) shape :  [b, n_seq],  -> [b*n_seq]

        # [ batch*n_seq, gnn_output]
        # GNN
        outputs_gnn, middle_feature, Aw = self.gnn.forward(batch_x)
        outputs_gnn = outputs_gnn.reshape([-1, self.n_seq, self.gnn_out_features])  # [b, n_seq, gnn_output]
        # diff
        pad_vec = outputs_gnn[:, 0, :]
        pad_vec = torch.unsqueeze(pad_vec, dim=1)
        vec = outputs_gnn[:, :-1, :]
        vec = torch.cat([pad_vec, vec], dim=1)  # [b, n_seq, gnn_output]
        diff_vec = outputs_gnn-vec
        diff_vec = diff_vec.reshape(-1, self.gnn_out_features)
        diff_out = F.leaky_relu(self.diff_mlp.forward(diff_vec))  # [b*n_seq, 1]
        diff_out = diff_out.reshape(-1)

        # Linear
        x_emb = F.leaky_relu(self.feature_linear(F.leaky_relu(outputs_gnn)))

        # RNN
        if self.rnn == 'transformer':
            seq_length = x_emb.shape[1]
            device = x_emb.device
            input_shape = x_emb.size()[:2]
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
            position_embeddings = self.emb_position(position_ids)
            x_emb = x_emb + position_embeddings
            outputs = self.rnn_model(x_emb)
            hiddens = None
        elif self.rnn == 'gru' or self.rnn == 'LSTM':
            outputs, hiddens = self.rnn_model(x_emb)
        else:
            outputs, hiddens = x_emb, None

        # CNN+pool
        # shape: <batch, nstep, features >  -> <batch, features>
        # lstm_output = F.adaptive_max_pool1d(outputs.transpose(1, 2), output_size=1).squeeze()
        if self.rnn_model is not None:
            conv_x_in = outputs.transpose(1, 2)
            conv_xs = []
            for convs in self.convs:
                conv_x = conv_x_in
                for conv in convs:
                    conv_x = conv(conv_x)
                    conv_x = F.relu(conv_x)
                conv_x = conv_x + conv_x_in
                conv_x = F.adaptive_avg_pool1d(conv_x, output_size=1).squeeze()
                conv_xs.append(conv_x)

            cnn_out = torch.cat(conv_xs, dim=1)
        else:
            conv_x = F.adaptive_avg_pool1d(outputs.transpose(1, 2), output_size=1).squeeze()
            cnn_out = conv_x  #
            # print(cnn_out.shape)
        # MLP
        # lstm_output = self.sumpool1(outputs_reshape).squeeze()
        x_outputs = F.relu(self.linear(cnn_out))
        x_outputs = self.linear2(x_outputs)

        if return_node_F:
            return x_outputs, outputs_gnn, outputs, Aw, middle_feature

        return x_outputs, outputs_gnn, outputs, Aw, diff_out
