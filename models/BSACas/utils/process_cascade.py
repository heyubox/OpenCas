import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import networkx as nx
import numpy as np
import pickle
import time
from tqdm import tqdm
import torch
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csc_matrix
from datetime import datetime
from config import opts
import scipy.sparse as sp

import math


class GraphData:
    def __init__(self, g, label=None, node_features=None, node_tags=None, edge_mat=None):
        self.label = label
        self.g = g
        # self.node_tags = node_tags
        self.node_features = node_features
        self.edge_mat = edge_mat
        self.max_neighbor = 0
        self.nodes_num = 0
        self.current_num = 0


def id2node_in_batch(graph_seq):
    id2nodes_new = {}
    graph_seq = graph_seq.reshape([-1])
    shift = 0
    for g in graph_seq:
        map_dict = g.id2nodes
        for k,v in map_dict.items():
            
            id2nodes_new[int(k)+shift]=str(v)
        shift+=len(map_dict)
    return id2nodes_new


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def count_cascade_num(file_name):
    num = 0
    for line in open(file_name):
        num += 1
    return num


def parse_line(parts):
    hour = None
    cascade_id = parts[0]
    # msg_time = int(parts[2])
    time_str = parts[2]
    try:
        msg_time = int(time_str)
        hour = time.strftime("%H", time.localtime(msg_time))
        hour = int(hour)
    except:
        try:
            ts = time.strptime(parts[2], '%Y-%m-%dT%H:%M:%S')
            hour = ts.tm_hour
        except:
            ts = time.localtime(int(parts[2]))
            hour = ts.tm_hour

    retweet_number = int(parts[3])
    paths = parts[4].split(' ')
    return cascade_id, msg_time, hour, retweet_number, paths


def parse_line_citation(parts):
    cascade_id = parts[0]
    msg_time = datetime.strptime(parts[2], '%Y-%m-%d')
    hour = msg_time.hour
    retweet_number = int(parts[3])
    paths = parts[4].split(' ')
    return cascade_id, msg_time, hour, retweet_number, paths


def get_observation_path(paths, observation_time, pred_time=3600 * 24, return_label=False):
    """
    get observation path during a time
    :param paths:
    :param observation_time:
    :return: list of [nodes, retweet_t]
    """
    observation_path = []
    # pred_time = 3600 * 24
    label = 0
    for p in paths:
        p_split = p.split(":")
        nodes = p_split[0].split('/')
        retweet_t = int(p_split[1])  # retweet time
        nodes_ok = True
        for n in nodes:
            if int(n) == -1:
                nodes_ok = False
        if not nodes_ok:
            print("error id at {}".format(nodes))
            continue

        if retweet_t < observation_time:
            observation_path.append([nodes, retweet_t])
        if retweet_t <= pred_time:
            label += 1

    observation_path = sorted(observation_path, key=lambda x: x[1])
    if return_label:
        return observation_path, label - len(observation_path)

    return observation_path


def get_observation_path_citation(paths, observation_time, pred_time=3600 * 24, return_label=False):
    """
    get observation path during a time
    :param paths:
    :param observation_time:
    :return: list of [nodes, retweet_t]
    """
    observation_path = []
    # pred_time = 3600 * 24
    label = 0
    for p in paths:
        p_split = p.split(":")
        nodes = p_split[0].split('/')
        retweet_t = int(p_split[1])  # retweet time
        nodes_ok = True
        for n in nodes:
            if int(n) == -1:
                nodes_ok = False
        if not nodes_ok:
            print(nodes)
            continue

        if retweet_t < observation_time:
            observation_path.append([nodes, retweet_t])
        if retweet_t <= pred_time:
            label += 1

    observation_path = sorted(observation_path, key=lambda x: x[1])
    if return_label:
        return observation_path, label - len(observation_path)

    return observation_path


def get_observation_path_steps(paths, observation_time, return_label=False):
    """
    get observation path during a time
    :param paths:
    :param observation_time:
    :return: list of [nodes, retweet_t]
    """
    observation_path = []
    pred_time = 3600 * 24
    label = 0
    for p in paths:
        p_split = p.split(":")
        nodes = p_split[0].split('/')
        retweet_t = int(p_split[1])  # retweet time
        nodes_ok = True
        for n in nodes:
            if int(n) == -1:
                nodes_ok = False
        if not nodes_ok:
            print(nodes)
            continue

        if retweet_t < observation_time:
            observation_path.append([nodes, retweet_t])
        if retweet_t <= pred_time:
            label += 1
    if return_label:
        return observation_path, label - len(observation_path)
    return observation_path


def qualified_cascade(file_name, observation_time, pred_time, viral=False):

    qualified_data_ids = {}

    # test on different tpye of networkx viral or non-viral
    if viral:
        viral_cid_avg_dist = pickle.load(open('viral_cid_avg_dist.pkl', 'rb'))
    cascades_total = 0
    cascades_valid_total = 0
    discard_midnight = 0
    discard_outer = 0
    num_lines = sum(1 for line in open(file_name, 'r'))
    for line in tqdm(open(file_name), total=num_lines):
        cascades_total += 1
        parts = line.split('\t')
        if len(parts) != 5:
            print('wrong format!')
            continue

        cacade_id, msg_time, hour, retweet_number, paths = parse_line(parts)

        if hour <= opts.start_hour or hour >= opts.end_hour:  # remove tweet published in midnight
            discard_midnight += 1
            continue

        if retweet_number != len(paths):
            print('wrong number of nodes', retweet_number, len(paths))

        observation_paths, label = get_observation_path(paths, observation_time, pred_time, return_label=True)

        if viral:
            if cacade_id not in viral_cid_avg_dist:
                continue
            if viral_cid_avg_dist[cacade_id] < 2.5:
                continue

        if len(observation_paths) < opts.least_num or len(observation_paths) > opts.up_num:  # or label>1000:
            discard_outer += 1
            continue
        else:
            cascades_valid_total += 1
            qualified_data_ids[cacade_id] = hour

    print("total_readin:", cascades_total)
    print("discard_midnight:", discard_midnight)
    print("discard_outer:", discard_outer)
    print('total valid:', cascades_valid_total)
    return qualified_data_ids


def qualified_cascade_citation(file_name, observation_time=365 * 5):
    qualified_data_ids = {}
    for line in tqdm(open(file_name), total=42186):
        parts = line.split('\t')
        if len(parts) != 5:
            print('wrong format!')
            continue

        cacade_id, msg_time, hour, retweet_number, paths = parse_line_citation(parts)

        # if hour <= 7 or hour >= 19:  # remove tweet published in midnight
        #     continue

        if retweet_number != len(paths) and retweet_number != len(paths) - 1:
            print('wrong number of nodes', retweet_number, len(paths))

        observation_paths = get_observation_path(paths, observation_time)

        if retweet_number < 10 or retweet_number > 1000:
            continue
        # if len(observation_paths) < 10 or len(observation_paths) > 1000:
        #     continue
        else:
            qualified_data_ids[cacade_id] = ""

    print("total qualified data:", len(qualified_data_ids))

    return qualified_data_ids


def one_hot_fun(x, num_class=None):
    if not num_class:
        num_class = np.max(x) + 1
    b = np.zeros((len(x), num_class))
    b[range(len(x)), x] = 1
    return b


def features_by_degs(degs, one_hot=False):
    # divide the degree into 10 categories ( features)
    # 0-10000
    # deg_to_features = np.array([3, 5, 10, 20, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000, 20000])
    # deg_to_features = np.array([1, 2, 3, 5, 8, 10, 20, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000, 20000])
    deg_to_features = np.array(list(range(1000)))
    if not one_hot:
        return np.searchsorted(deg_to_features, degs)

    return one_hot_fun(np.searchsorted(deg_to_features, degs), num_class=len(deg_to_features))


def features_by_degs_sparse_old(g, feature_len=100, feature_log=False):
    # without mapping and scaling the dgerees
    degs = dict(nx.degree(g))
    row = np.array(list(degs.keys()))
    col = np.array(list(degs.values()))
    values = np.ones(len(row))
    if feature_log:
        indexes = np.linspace(0, 10, feature_len)
        col = np.log2(col + 1)
        col = np.searchsorted(indexes, col)
        col[col > 99] = feature_len - 1  # incase
    node_features = csc_matrix((values, (row, col)), shape=[len(values), feature_len])
    return node_features


def features_by_degs_sparse(g, feature_len=100, max_deg=1000):
    # mapping the degree
    degs = dict(nx.degree(g))
    row = np.array(list(degs.keys()))
    col = np.array(list(degs.values()))
    values = np.ones(len(row))

    half_feature = feature_len / 2
    index = np.argwhere(col > half_feature).squeeze()
    col[index] = np.around((col[index] - half_feature) * half_feature / (max_deg - half_feature) + half_feature)
    col[col >= feature_len] = feature_len - 1

    node_features = csc_matrix((values, (row, col)), shape=[len(values), feature_len])
    return node_features


def features_fourier_transformation(g, feature_len=100, l_max=2.0, directed=True):
    if len(g) == 1:
        # return np.zeros(shape=[1, feature_len])
        node_features = np.zeros(shape=[1, feature_len])
        node_features[0, 0] = 1.0
        return csc_matrix(node_features)
    try:
        # L = nx.directed_laplacian_matrix(g)  # seems like L has already normalized
        L = nx.laplacian_matrix(g).toarray()
    except Exception as e:
        # print(e)
        if len(g) == 2:
            L = nx.laplacian_matrix(g.to_undirected()).toarray()
            # return np.zeros(shape=[2, feature_len])
        print(g)
    # I = np.identity(L.shape[0])
    # L_scale = L - I
    L_scale = L
    eig_values, eig_vectors = np.linalg.eigh(L_scale)
    node_features = np.array(eig_vectors)
    if feature_len > len(eig_values):
        node_features = node_features.transpose()
        pad_num = feature_len - len(eig_values)
        N = len(eig_values)
        paddings = np.zeros(shape=[N, pad_num])
        node_features = np.concatenate([node_features, paddings], axis=-1)
        # return node_features
        return csc_matrix(node_features)
    else:
        # return node_features[-feature_len:, :].transpose()
        return csc_matrix(node_features[-feature_len:, :].transpose())


def contruct_graph(observation_paths, feature_method='new', directed=False, label=0, feature_len=100, delete_graph=True):
    start = time.time()
    nodes = []
    [nodes.extend(ns) for ns, retweet_time in observation_paths]
    nodes = list(set(nodes))
    nodes2id = dict(zip(nodes, range(len(nodes))))
    # nodes2id = dict(zip(nodes, nodes))

    g = nx.Graph()
    directed = False
    if directed:
        g = nx.DiGraph()  #

    g.add_node(nodes2id[observation_paths[0][0][0]])  # add root node

    for ns, retweet_time in observation_paths:
        for i, n in enumerate(ns[1:]):
            edge = nodes2id[ns[i]], nodes2id[ns[i + 1]]
            g.add_edge(edge[0], edge[1])
    # print('construct graph', time.time() - start)
    feature_log = False
    if feature_method == 'new':
        node_features = features_by_degs_sparse(g, feature_len=feature_len)
        # node_features = features_fourier_transformation(g, feature_len=feature_len)
    else:
        node_features = features_by_degs_sparse_old(g, feature_len=feature_len, feature_log=feature_log)

    edge_mat = construct_edge_matrix(g)
    g1 = GraphData(g=None, node_features=node_features, edge_mat=edge_mat)
    g1.label = label
    g1.id2nodes = dict(zip(nodes2id.values(), nodes2id.keys()))
    g1.current_num = len(g)
    g1.nodes_num = len(g)
    if delete_graph:
        del g
    return g1


def construct_edge_matrix(g):
    if len(g.edges()) == 0:
        edge_mat = torch.LongTensor([[0, 0]]).transpose(0, 1)
    else:
        edges = [[i, j] for i, j in g.edges()]
        edges.extend([[j, i] for i, j in g.edges()])
        edge_mat = torch.LongTensor(edges).transpose(0, 1)

    return edge_mat

    # if not g.is_directed():
    #     edges = list(g.to_directed().edges())
    # else:
    #     edges = list(g.edges())
    # return torch.LongTensor(edges)

    # g_sparse = nx.to_scipy_sparse_matrix(g)
    # return g_sparse


def parse_observation_path(observation_paths, label, directed=False, feature_len=100, delete_graph=True, feature_method='new'):

    # use degree in the cascade
    g = contruct_graph(observation_paths, feature_method=feature_method, directed=directed, label=label, feature_len=feature_len, delete_graph=delete_graph)

    # g.current_num = len(observation_paths)

    # parse graph edge matrix
    start = time.time()

    # print('edge mat', time.time()-start)
    return g


def parse_sequence_counts(observation_paths, interval=180, observation_time=3600):
    # intervals = np.array([(i+1)*interval for i in range(int(observation_time/interval))])
    intervals = np.array([(i + 1) * interval for i in range(math.ceil(observation_time / interval))])
    observation_paths_sorted = sorted(observation_paths, key=lambda x: x[1])
    retweet_times = [path[1] for path in observation_paths_sorted]
    retweet_intervals = np.searchsorted(intervals, retweet_times)
    values, counts = np.unique(retweet_intervals, return_counts=True)
    retweet_couts = np.zeros(len(intervals))
    retweet_couts[values] = counts

    return np.cumsum(retweet_couts)


def process_qualified_cascade(qualified_data_ids, file_name, observation_time=3600, feature_len=100):
    g_list = []
    num = 0
    for line in tqdm(open(file_name), ):
        parts = line.split('\t')
        cascade_id, msg_time, hour, retweet_number, paths = parse_line(parts)
        if cascade_id not in qualified_data_ids:
            continue
        observation_paths, label = get_observation_path(paths, observation_time, return_label=True)

        g = parse_observation_path(observation_paths, label, feature_len=feature_len)
        # pickle.dump(g, open('laplacian/laplacian_feature_{}'.format(cascade_id), 'wb'))
        g_list.append(g)
        num += 1

    return g_list


def split_data(g_list, labels, fold_idx=0, method='Kfold'):
    g_list_np = np.array(g_list)
    if method == 'Kfold':
        skf = StratifiedKFold(10, shuffle=True, random_state=0)
        index_list = []
        for train_index, test_index in skf.split(range(len(g_list)), labels):
            index_list.append([train_index, test_index])
    else:
        index_list = []
        for i in range(10):
            train_index = np.random.choice(list(range(len(labels))), replace=False, size=int(len(labels) * 0.9))
            test_index = list(set(range(len(labels))) - set(train_index))
            index_list.append([train_index, test_index])

    train_index, test_index = index_list[fold_idx]
    X_train, Y_train = g_list_np[train_index], labels[train_index]
    X_test, Y_test = g_list_np[test_index], labels[test_index]
    return X_train, X_test, Y_train, Y_test  # attention


def load_observation_paths(file_name, qualified_data_ids, observation_time, input_features=100, path=None):
    if path is None:
        path = ""
    observation_paths_file = path + '/observation_paths_{}.pkl'.format(observation_time)
    label_file = path + '/labels_{}.pkl'.format(observation_time)

    if not os.path.exists(observation_paths_file):
        observation_paths_total = []
        labels = []
        num_lines = sum(1 for line in open(file_name, 'r'))

        for line in tqdm(open(file_name), total=num_lines):

            parts = line.split('\t')
            cascade_id, msg_time, hour, retweet_number, paths = parse_line(parts)
            if cascade_id not in qualified_data_ids:
                continue
            observation_paths, label = get_observation_path(paths, observation_time, return_label=True)
            labels.append(label)
            observation_paths_total.append(observation_paths)

        pickle.dump(observation_paths_total, open(observation_paths_file, 'wb'))
        pickle.dump(labels, open(label_file, 'wb'))

    else:
        observation_paths_total = pickle.load(open(observation_paths_file, 'rb'))
        labels = pickle.load(open(label_file, 'rb'))

    return observation_paths_total, labels


def get_nodes_seq(observation_paths, interval_index, nodes):

    nodes2id = dict(zip(nodes, range(len(nodes))))

    nodes_seq = []
    nodes2id_tmp = {}
    nodes = []

    for i, index in enumerate(interval_index):
        if i == 0:
            for ns, t in observation_paths[:int(index)]:
                for n in ns:
                    if n not in nodes2id_tmp:
                        nodes.append(nodes2id[n])
                        nodes2id_tmp[n] = ""
        else:
            for ns, t in observation_paths[interval_index[i - 1]:int(index)]:
                for n in ns:
                    if n not in nodes2id_tmp:
                        nodes.append(nodes2id[n])
                        nodes2id_tmp[n] = ""

        nodes_seq.append(nodes.copy())
    return nodes_seq


def get_deg_sequence(observation_paths, observation_time, interval, label, interval_index, feature_method='new', feature_len=100):
    # g_seq = []
    # for i, index in enumerate(interval_index):
    #     g = parse_observation_path(observation_paths[:int(index)], label, directed=False, feature_len=50, delete_graph=False)
    #     g_seq.append(g)
    from scipy.sparse import csc_matrix
    interval_index = interval_index.astype(np.int32)
    # g = parse_observation_path(observation_paths[:int(interval_index[-1])], label, directed=False, feature_len=100, delete_graph=False, feature_method='new')

    nodes = []
    # [nodes.extend(ns) for ns, retweet_time in observation_paths]
    # nodes = list(set(nodes))

    from collections import OrderedDict
    nodes = OrderedDict()
    nodes[observation_paths[0][0][0]] = 0

    n_node = 1
    for ns, retweet_time in observation_paths[1:]:  # sorted by time
        for node in ns[1:]:
            if node not in nodes:
                nodes[node] = n_node
                n_node += 1

    nodes = list(nodes.keys())

    nodes2id = dict(zip(nodes, range(len(nodes))))

    g = nx.Graph()
    g.add_node(nodes2id[observation_paths[0][0][0]])  # add root node

    for ns, retweet_time in observation_paths:
        for i, n in enumerate(ns[1:]):
            edge = nodes2id[ns[i]], nodes2id[ns[i + 1]]
            g.add_edge(edge[0], edge[1])
    # print('construct graph', time.time() - start)

    # feature_len = 100
    # feature_method = 'old'
    if feature_method == 'new':
        node_features = features_by_degs_sparse(g, feature_len=feature_len)
    else:
        node_features = features_by_degs_sparse_old(g, feature_len=feature_len, feature_log=True)
    # nodes = []
    # [nodes.extend(ns) for ns, retweet_time in observation_paths]
    # nodes = list(set(nodes))
    # nodes2id = dict(zip(nodes, range(len(nodes))))
    #
    #
    # feature_seq = []
    # for index in interval_index:
    #     nodes_g =[]
    #     [nodes_g.extend(ns) for ns, retweet_time in observation_paths[:index]]
    #     nodes_g = list(set(nodes_g))
    #     nodes_g = [nodes2id[n] for n in nodes_g]
    #     features = g.node_features[nodes_g]
    #     feature_seq.append(features.sum(axis=0))

    nodes_seq = get_nodes_seq(observation_paths, interval_index, nodes)
    # faster way:
    feature_seq = []
    for i in range(len(interval_index)):
        feature_seq.append(node_features[nodes_seq[i]].sum(axis=0).tolist()[0])
    return feature_seq


def get_graph_sequence(observation_paths, observation_time, interval, label, interval_index, feature_method='new', feature_len=100, directed=False):
    """
    get graph sequence for GNNLSTM / GRU
    :param observation_paths:
    :param observation_time:
    :param interval:
    :param label:
    :param interval_index:
    :param feature_method:
    :param feature_len:
    :return:
    """

    interval_index = interval_index.astype(np.int32)

    # nodes = {}
    from collections import OrderedDict
    nodes = OrderedDict()
    # [nodes.extend(ns) for ns, retweet_time in observation_paths] # ignore time and sorted
    nodes[observation_paths[0][0][0]] = 0

    n_node = 1
    for ns, retweet_time in observation_paths[1:]:  # sorted by time
        for node in ns[1:]:
            if node not in nodes:
                nodes[node] = n_node
                n_node += 1

    # nodes = list(set(nodes))
    nodes = list(nodes.keys())
    nodes2id = dict(zip(nodes, range(len(nodes))))

    g = nx.Graph()
    g.add_node(nodes2id[observation_paths[0][0][0]])  # add root node

    for ns, retweet_time in observation_paths:
        for i, n in enumerate(ns[1:]):
            edge = nodes2id[ns[i]], nodes2id[ns[i + 1]]
            if edge[0] != edge[1]:
                g.add_edge(edge[0], edge[1])
    # print('construct graph', time.time() - start)

    # feature_len = 100
    # feature_method = 'new'
    if feature_method == 'new':
        node_features = features_by_degs_sparse(g, feature_len=feature_len)
    else:
        node_features = features_by_degs_sparse_old(g, feature_len=feature_len, feature_log=True)

    # nodes = []
    # [nodes.extend(ns) for ns, retweet_time in observation_paths]
    # nodes = list(set(nodes))
    # nodes2id = dict(zip(nodes, range(len(nodes))))
    #
    #
    # feature_seq = []
    # for index in interval_index:
    #     nodes_g =[]
    #     [nodes_g.extend(ns) for ns, retweet_time in observation_paths[:index]]
    #     nodes_g = list(set(nodes_g))
    #     nodes_g = [nodes2id[n] for n in nodes_g]
    #     features = g.node_features[nodes_g]
    #     feature_seq.append(features.sum(axis=0))

    nodes_seq = get_nodes_seq(observation_paths, interval_index, nodes)
    # faster way:

    graph_seq = []
    for i in range(len(interval_index)):
        # this is for equal formulation test
        # node_feature = csc_matrix(node_features[nodes_seq[i]].sum(axis=0).tolist()[0])
        node_feature = node_features[nodes_seq[i]]

        edge_list = list(g.subgraph(nodes_seq[i]).edges())
        if len(edge_list) == 0:
            edge_mat = np.array([[0], [0]])
        else:
            edge_mat = np.array(edge_list).transpose()
            # bi-direction
            edge_list_reverse = edge_mat[::-1]
            edge_mat = np.concatenate([edge_mat, edge_list_reverse], axis=-1)

            # if edge_mat needs to be Tensor
            edge_mat = torch.LongTensor(edge_mat)

        # if use chebnet

        g1 = GraphData(g=None, node_features=node_feature, edge_mat=edge_mat)
        g1.label = label
        g1.nodes_num = len(nodes_seq[i])
        g1.current_num = len(nodes_seq[i])
        graph_seq.append(g1)
    return graph_seq


def get_graph_laplacian(observation_paths, observation_time, interval, label, interval_index, feature_method='new', feature_len=100):
    interval_index = interval_index.astype(np.int32)

    # nodes = {}
    from collections import OrderedDict
    nodes = OrderedDict()
    # [nodes.extend(ns) for ns, retweet_time in observation_paths] # ignore time and sorted
    nodes[observation_paths[0][0][0]] = 0

    n_node = 1
    for ns, retweet_time in observation_paths[1:]:  # sorted by time
        for node in ns[1:]:
            if node not in nodes:
                nodes[node] = n_node
                n_node += 1

    # nodes = list(set(nodes))
    nodes = list(nodes.keys())
    nodes2id = dict(zip(nodes, range(len(nodes))))

    # g = nx.DiGraph()
    g = nx.Graph()
    g.add_node(nodes2id[observation_paths[0][0][0]])  # add root node

    if len(nodes) <= 2:
        return []

    for ns, retweet_time in observation_paths:
        for i, n in enumerate(ns[1:]):
            edge = nodes2id[ns[i]], nodes2id[ns[i + 1]]
            if edge[0] != edge[1]:
                g.add_edge(edge[0], edge[1])

    nodes_order = g.nodes()

    # print('construct graph', time.time() - start)

    # feature_len = 100
    # feature_method = 'new'
    if feature_method == 'new':
        node_features = features_by_degs_sparse(g, feature_len=feature_len)
        # node_features = features_fourier_transformation(g, feature_len=feature_len)
    else:
        node_features = features_by_degs_sparse_old(g, feature_len=feature_len, feature_log=True)

    # nodes = []
    # [nodes.extend(ns) for ns, retweet_time in observation_paths]
    # nodes = list(set(nodes))
    # nodes2id = dict(zip(nodes, range(len(nodes))))
    #
    #
    # feature_seq = []
    # for index in interval_index:
    #     nodes_g =[]
    #     [nodes_g.extend(ns) for ns, retweet_time in observation_paths[:index]]
    #     nodes_g = list(set(nodes_g))
    #     nodes_g = [nodes2id[n] for n in nodes_g]
    #     features = g.node_features[nodes_g]
    #     feature_seq.append(features.sum(axis=0))

    nodes_seq = get_nodes_seq(observation_paths, interval_index, nodes)
    # faster way:

    graph_seq = []
    for i in range(len(interval_index)):
        # this is for equal formulation test
        # node_feature = csc_matrix(node_features[nodes_seq[i]].sum(axis=0).tolist()[0])
        node_feature = node_features[nodes_seq[i]]

        edge_list = list(g.subgraph(nodes_seq[i]).edges())
        if len(edge_list) == 0:
            edge_mat = np.array([[0], [0]]).transpose()
        else:
            edge_mat = np.array(edge_list)  # .transpose()

            # bi-direction
            edge_list_reverse = edge_mat[::-1]
            edge_mat = np.concatenate([edge_mat, edge_list_reverse], axis=-1)

        edge_mat = torch.LongTensor(edge_mat)
        # if use chebnet

        g1 = GraphData(g=None, node_features=node_feature, edge_mat=edge_mat)
        g1.label = label
        g1.nodes_num = len(nodes_seq[i])
        # g1.nodes_num = node_feature.shape[0]
        g1.current_num = node_feature.shape[0]
        graph_seq.append(g1)

    return graph_seq


# def load_observation_paths(file_name, qualified_data_ids, observation_time):
#     num = 0
#     if not os.path.exists('paths/observation_paths_{}.pkl'.format(observation_time)):
#         observation_paths_total = []
#         labels = []
#         for line in tqdm(open(file_name), total=42186):
#
#             num += 1
#             if num % 1000 ==0 :
#                 print(num)
#
#             parts = line.split('\t')
#             cascade_id, msg_time, hour, retweet_number, paths = parse_line(parts)
#             if cascade_id not in qualified_data_ids:
#                 continue
#             observation_paths, label = get_observation_path(paths, observation_time, return_label=True)
#             labels.append(label)
#             observation_paths_total.append(observation_paths)
#
#         pickle.dump(observation_paths_total, open('paths/observation_paths_{}.pkl'.format(observation_time), 'wb'))
#         pickle.dump(labels, open('paths/labels_{}.pkl'.format(observation_time), 'wb'))
#
#     else:
#         observation_paths_total = pickle.load(open('paths/observation_paths_{}.pkl'.format(observation_time), 'rb'))
#         labels = pickle.load(open('paths/labels_{}.pkl'.format(observation_time), 'rb'))
#
#     return observation_paths_total, labels

if __name__ == '__main__':
    # num = count_cascade_num(file_name)
    # print(num)
    file_name = opts.RAWDATA_PATH
    qualified_data_ids = qualified_cascade(file_name, opts.observation_time, opts.prediction_time)
    # 68989
    # g_list = process_qualified_cascade(qualified_data_ids, file_name, observation_time, feature_len=100)
    # print('total built grpahs:{}'.format(len(g_list)))
