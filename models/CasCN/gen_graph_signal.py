from tqdm import tqdm
import time
import gc
import scipy.sparse
import utils.caslaplacian as caslaplacian
import networkx as nx
from config import opts
import numpy as np
import pickle
import sys
sys.path.append('./')
# from preprocessing import caslaplacian
LABEL_NUM = 0


# trans the original ids to 1~n
class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)

# trainsform the sequence to list


def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = []  # walk[0] = cascadeID
            cascade_list = []
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0]  # node
                t = walks[i].split(":")[1]  # time

                cascade_list.append([[str(xx) for xx in s.split(",")], int(t)])
            cascade_list.sort(key=lambda x: x[1])
            graphs[walks[0]] = cascade_list[:100]
    return graphs

# read label and size from cascade file


def read_labelANDsize(filename):
    labels = {}
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            labels[profile[0]] = profile[-1]
            sizes[profile[0]] = int(profile[3])
    return labels, sizes


def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print("length of original isd:", len(original_ids))
    return original_ids


def get_nodes(graph):
    nodes = {}
    j = 0
    for walk in graph:
        for i in walk[0]:
            if i not in nodes.keys():
                nodes[i] = j
                j = j+1
    return nodes


def write_XYSIZE_data(graphs, labels, sizes, LEN_SEQUENCE, NUM_SEQUENCE, index, max_num, filename):
    # get the x,y,and size  data
    id_data = []
    x_data = []
    y_data = []
    sz_data = []
    time_data = []
    Laplacian_data = []
    graphs_info = []
    for key, graph in tqdm(graphs.items()):
        graph_info = []
        for g in graph:
            graph_info.append(g[0][-1])
        graphs_info.append(graph_info)
        id = key
        label = labels[key].split()
        y = int(label[LABEL_NUM])  # label
        temp = []
        temp_time = []  # store time
        size_temp = len(graph)
        # if size_temp != sizes[key]:
        #     print(size_temp, sizes[key])
        nodes_items = get_nodes(graph)
        nodes_list = nodes_items.values()
        nx_G = nx.DiGraph()
        nx_G.add_nodes_from(nodes_list)
        for walk in graph:
            walk_time = walk[1]
            temp_time.append(walk_time)
            if walk_time == 0:
                nx_G.add_edge(nodes_items.get(walk[0][0]), nodes_items.get(walk[0][0]))
            for i in range(len(walk[0])-1):
                nx_G.add_edge(nodes_items.get(walk[0][i]), nodes_items.get(walk[0][i+1]))
            temp_adj = nx.to_pandas_adjacency(nx_G)
            N = len(nx_G)
            # if len(nx_G.edges()) > 100:
            #     print('larger than 100')
            if N < max_num:
                col_padding = np.zeros(shape=(N, max_num - N))
                A_col_padding = np.column_stack((temp_adj, col_padding))
                row_padding = np.zeros(shape=(max_num - N, max_num))
                A_col_row_padding = np.row_stack((A_col_padding, row_padding))
                temp_adj = scipy.sparse.coo_matrix(A_col_row_padding, dtype=np.float32)
            else:
                temp_adj = scipy.sparse.coo_matrix(temp_adj, dtype=np.float32)
            temp.append(temp_adj)
        # caculate laplacian
        L = caslaplacian.calculate_scaled_laplacian_dir(nx_G, lambda_max=None)
        M, M = L.shape
        M = int(M)
        L = L.todense()
        if M < max_num:
            col_padding_L = np.zeros(shape=(M, max_num - M))
            L_col_padding = np.column_stack((L, col_padding_L))
            row_padding = np.zeros(shape=(max_num - M, max_num))
            L_col_row_padding = np.row_stack((L_col_padding, row_padding))
            Laplacian = scipy.sparse.coo_matrix(L_col_row_padding, dtype=np.float32)
        else:
            Laplacian = scipy.sparse.coo_matrix(L, dtype=np.float32)
        if len(temp) < NUM_SEQUENCE:
            zero_padding = np.zeros(shape=(max_num, max_num))
            zero_padding = scipy.sparse.coo_matrix(zero_padding, dtype=np.float32)
            for i in range(NUM_SEQUENCE-len(temp)):
                temp.append(zero_padding)
                i = i+1
        time_data.append(temp_time)
        id_data.append(id)
        x_data.append(temp)
        y_data.append(np.log(y+1.0)/np.log(2.0))
        Laplacian_data.append(Laplacian)
        sz_data.append(size_temp)
    gc.collect()
    pickle.dump((id_data, x_data, Laplacian_data, y_data, sz_data, time_data, index.length(), graphs_info), open(filename, 'wb'))


def get_maxsize(sizes):
    max_size = 0
    for cascadeID in sizes:
        max_size = max(max_size, sizes[cascadeID])
    gc.collect()
    return max_size


def get_max_length(graphs):
    len_sequence = 0
    max_num = 0
    for cascadeID in graphs:
        max_num = max(max_num, len(graphs[cascadeID]))
        for sequence in graphs[cascadeID]:
            len_sequence = max(len_sequence, len(sequence[0]))
    gc.collect()
    return len_sequence


def get_max_node_num(graphs):
    max_num = 0
    for key, graph in graphs.items():
        nodes = get_nodes(graph)
        max_num = max(max_num, len(nodes))
    return max_num


if __name__ == "__main__":

    ### data set ###
    graphs_train = sequence2list(opts.shortestpath_train)
    graphs_val = sequence2list(opts.shortestpath_val)
    graphs_test = sequence2list(opts.shortestpath_test)

    ## get Laplacian ##
    cascade_train = opts.cascade_train
    cascade_test = opts.cascade_test
    cascade_val = opts.cascade_val

    ### get labels ###
    labels_train, sizes_train = read_labelANDsize(opts.cascade_train)  # 标签以及观测到的长度
    labels_val, sizes_val = read_labelANDsize(opts.cascade_val)
    labels_test, sizes_test = read_labelANDsize(opts.cascade_test)
    NUM_SEQUENCE = 100  # max(get_maxsize(sizes_train), get_maxsize(sizes_val), get_maxsize(sizes_test))  # 观测到的最大长度

    print("NUM_SEQUENCE:", NUM_SEQUENCE)
    print('Train:{} Test:{} Valid:{}'.format(len(labels_train), len(labels_test), len(labels_val)))

    LEN_SEQUENCE_train = get_max_length(graphs_train)  # 最大步长
    LEN_SEQUENCE_val = get_max_length(graphs_val)
    LEN_SEQUENCE_test = get_max_length(graphs_test)
    LEN_SEQUENCE = 100  # max(LEN_SEQUENCE_train, LEN_SEQUENCE_val, LEN_SEQUENCE_test)

    max_num_train = get_max_node_num(graphs_train)
    max_num_test = get_max_node_num(graphs_test)
    max_num_val = get_max_node_num(graphs_val)
    max_num = 100  # max(max_num_train, max_num_test, max_num_val)

    # get the total original_ids and tranform the index from 0 ~n-1
    original_ids = get_original_ids(graphs_train)\
        .union(get_original_ids(graphs_val))\
        .union(get_original_ids(graphs_test))

    original_ids.add(-1)
    # index is new index
    index = IndexDict(original_ids)

    print('max node: ', max_num)

    print("create train")

    start = time.time()

    def dict_slice(adict, start=0, end=100):
        keys = list(adict.keys())
        dict_slice = {}
        for k in keys[start:end]:
            dict_slice[k] = adict[k]
        return dict_slice

    write_XYSIZE_data(graphs_train, labels_train, sizes_train, LEN_SEQUENCE, NUM_SEQUENCE, index, max_num, opts.train_pkl)
    print(time.time() - start)
    print("create val and test")
    write_XYSIZE_data(graphs_val, labels_val, sizes_val, LEN_SEQUENCE, NUM_SEQUENCE, index, max_num, opts.val_pkl)
    write_XYSIZE_data(graphs_test, labels_test, sizes_test, LEN_SEQUENCE, NUM_SEQUENCE, index, max_num, opts.test_pkl)
    pickle.dump((len(original_ids), NUM_SEQUENCE, LEN_SEQUENCE, max_num), open(opts.information, 'wb'))

    print('NUM_SEQUENCE:', NUM_SEQUENCE,  'LEN_SEQUENCE:', LEN_SEQUENCE)
    print(time.time()-start)
    print("Finish!!!")
