import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
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
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, device, return_middle_feature=False,
                 mode='GPN', return_tsne=False):
        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.return_middle_feature = return_middle_feature
        self.return_tsne = return_tsne

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        # change output_dim to middle_dim = 64
        middle_dim = 32
        self.linears_prediction = torch.nn.ModuleList()
        self.mode = mode
        if self.mode == 'GIN':  # otherwise GPN
            for layer in range(num_layers):
                if layer==0:
                    self.linears_prediction.append(nn.Linear(input_dim, output_dim))
                else:
                    self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        else:
            for layer in range(num_layers):
                if layer == 0:
                    self.linears_prediction.append(nn.Linear(input_dim, middle_dim))
                else:
                    self.linears_prediction.append(nn.Linear(hidden_dim, middle_dim))

            # self.output_layer = nn.Linear(middle_dim, output_dim)
            self.output_layer = nn.Linear(middle_dim, middle_dim)
            self.output_layer2 = nn.Linear(middle_dim, output_dim)


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            # start_idx.append(start_idx[i] + len(graph.g))
            start_idx.append(start_idx[i] + graph.current_num)
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            # start_idx.append(start_idx[i] + len(graph.g))
            start_idx.append(start_idx[i] + graph.current_num) # modified
            # if len(graph.edge_mat) == 0:  # if the edge_mat is not .t() before
            #     edge_mat_list.append(torch.LongTensor([[0, 0]]).t() + start_idx[i])
            # else:
            #     edge_mat_list.append(graph.edge_mat.t() + start_idx[i])

            if len(graph.edge_mat) == 0:  # if the preprocess already .t()
                edge_mat_list.append(torch.LongTensor([[0, 0]]) + start_idx[i])
            else:
                edge_mat_list.append(graph.edge_mat + start_idx[i])

        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            # start_idx.append(start_idx[i] + len(graph.g))
            start_idx.append(start_idx[i] + graph.current_num) # modified

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                # elem.extend([1./len(graph.g)]*len(graph.g))
                elem.extend([1. / graph.current_num] * graph.current_num)

            else:
            ###sum pooling
                # elem.extend([1]*len(graph.g))
                elem.extend([1] * graph.current_num)

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)


    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            if isinstance(h, torch.sparse.LongTensor) or isinstance(h, torch.sparse.FloatTensor):  #add
                h = h.to_dense().type(torch.float32)
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    def array2tensor(self, node_features):

        rows, cols = node_features.nonzero()
        data = node_features.data
        # return torch.sparse.LongTensor(torch.LongTensor(np.vstack([rows, cols])), torch.LongTensor(data), size=(node_features.shape))
        # attention in Fourier transformation, feature type is float
        return torch.sparse.LongTensor(torch.LongTensor(np.vstack([rows, cols])), torch.FloatTensor(data), size=(node_features.shape))


    def forward(self, batch_graph, x_concat=None):
        # X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        # X_concat = torch.cat([self.array2tensor(graph.node_features) for graph in batch_graph], 0).to(self.device)
        import time
        start = time.time()
        # print(start)
        if x_concat is not None:
            X_concat = x_concat.to(self.device)
        else:
            X_concat = torch.cat([self.array2tensor(graph.node_features).to_dense() for graph in batch_graph]).to(self.device)
        # print(time.time()-start)
        start = time.time()
        graph_pool = self.__preprocess_graphpool(batch_graph)
        # print(time.time()-start)
        start = time.time()
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)
        # print(time.time()-start)
        start = time.time()
        #list of hidden representation at each layer (including input)
        # hidden_rep = [X_concat.to_dense().type(torch.float32)]
        hidden_rep = [X_concat.type(torch.float32)]

        # if isinstance(X_concat, torch.sparse.LongTensor) or isinstance(X_concat, torch.sparse.FloatTensor):
        #     X_concat = X_concat.to_dense().type(torch.float32)

        h = X_concat.type(torch.float32)

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)

        # print(time.time()-start)
        start = time.time()

        score_over_layer = 0
    
        #perform pooling over all nodes in each graph in every layer
        pool_total = []
        for layer, h in enumerate(hidden_rep):

            # only use first layer and last layer in GPN, just for testing
            if not self.mode == 'GIN':
                if not (layer == 0 or layer == len(hidden_rep)-1):
                    continue

            # add
            if isinstance(h, torch.sparse.LongTensor) or isinstance(h, torch.sparse.FloatTensor):
                h = h.to_dense()
            pooled_h = torch.spmm(graph_pool, h)
            pool_total.append(pooled_h)

            if self.mode == 'GIN':
                # score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                #                               training=self.training)
                score_over_layer += self.linears_prediction[layer](pooled_h)
            else:
                score_over_layer += self.linears_prediction[layer](pooled_h)


        # print(time.time()-start)
        start = time.time()

        if self.mode=='GIN':
            return score_over_layer

        # add an output layer
        # score_over_layer = F.relu(F.dropout(score_over_layer, 0.15, self.training))
        score_over_layer = F.relu(F.dropout(score_over_layer, 0.0, self.training))
        # outputs = F.leaky_relu(self.output_layer(score_over_layer))
        outputs = self.output_layer(score_over_layer)
        outputs = self.output_layer2(F.relu(outputs))
        # return score_over_layer, torch.cat(pool_total, dim=0)
        if self.return_middle_feature:
            return outputs, score_over_layer  # score_over_layer (batch, middle_hidden=32)

        if self.return_tsne:
            return outputs, score_over_layer, h

        # print(time.time()-start)
        # print('-----')
        start = time.time()
        return outputs



