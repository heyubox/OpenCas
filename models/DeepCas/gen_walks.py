"""
Generate random walk paths for each cascade graph and pre-train node embeddings.

Adapted from node2vec [1].

[1] node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016

"""
from typing import cast
from config import opts
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec
import time
import utils.node2vec as node2vec
import networkx as nx
import os
import numpy as np
import sys

sys.path.append('./')

data_path = os.path.join(opts.data_root, opts.dataset)
global_graph_file = opts.global_graph_file
cascade_file_prefix = os.path.join(data_path, "cascade_")
graph_walk_prefix = os.path.join(data_path, "random_walks_")
embed_prefix = os.path.join(data_path, "node_vec_")
# sets = ["train", "val", "test"]
# sets = ['train_test', 'test_test', 'val_test']
# only have records before observation time.
sets = ['train_new', 'test_new', 'val_new']

# exit(0)
node_to_degree = dict()
edge_to_weight = dict()
pseudo_count = 0.01


def get_global_info():
    rfile = open(global_graph_file, 'r')
    for line in rfile:
        line = line.rstrip('\r\n')
        parts = line.split("\t\t")
        # source = long(parts[0])
        source = int(parts[0])
        if parts[1] != "null":
            node_freq_strs = parts[1].split("\t")
            for node_freq_str in node_freq_strs:
                node_freq = node_freq_str.split(":")
                # I set weight=1.0 for all, you can change
                weight = int(node_freq[1])
                weight = 1.
                target = int(node_freq[0])
                #target = long(node_freq[0])
                if opts.trans_type == 0:
                    edge_to_weight[(source, target)] = weight
            degree = len(node_freq_strs)
        else:
            degree = 0
        node_to_degree[source] = degree
    rfile.close()
    return


def get_global_info_new():
    # in new implementation, save graph in edge list format (u, v, weight)
    # read it using networkx which is pretty convenient
    G = nx.read_edgelist(global_graph_file)
    for n in G:
        node_to_degree[n] = G.degree(n)
    return


def get_global_degree(node):
    return node_to_degree.get(node, 0)


def get_edge_weight(source, target):
    return edge_to_weight.get((source, target), 0)


def parse_graph(graph_string):
    parts = graph_string.split("\t")
    edge_strs = parts[4].split(" ")

    node_to_edges = dict()
    edge_strs = sorted(edge_strs, key=lambda x: int(x.split(":")[2]))

    # judge whether drop the dataset, the same with deephawkes
    observation_num = 0
    for i, edge_str in enumerate(edge_strs):
        edge_parts = edge_str.split(":")
        t = int(edge_parts[2])

        if opts.time_or_number == 'Num' and t > opts.observation_num + 1.:
            break

        if opts.time_or_number == 'Time' and t >= opts.observation_time:
            break

        observation_num += 1

    least_rewteetnum = opts.least_num  # 5 or 10
    if observation_num < least_rewteetnum:
        print("some erros happended before")
        return None, None, None

    prediction_time = opts.prediction_time
    predict_num = 0
    for i, edge_str in enumerate(edge_strs):
        edge_parts = edge_str.split(":")
        t = int(edge_parts[2])
        if t < prediction_time:
            predict_num += 1
    label = predict_num - observation_num

    for i, edge_str in enumerate(edge_strs):
        edge_parts = edge_str.split(":")
        source = int(edge_parts[0])
        target = int(edge_parts[1])

        # This is my Add, to limit the walks within observation time or number
        t = int(edge_parts[2])

        # when t=0, it is the root, so we need add one
        if opts.time_or_number == 'Num' and t > opts.observation_num:  # stop by observation number,
            break

        if opts.time_or_number == 'Time' and t > opts.observation_time:  # stop by observation time,
            break

        # source = long(edge_parts[0])
        # target = long(edge_parts[1])

        if not source in node_to_edges:
            neighbors = list()
            node_to_edges[source] = neighbors
        else:
            neighbors = node_to_edges[source]
        neighbors.append((target, get_global_degree(target)))

    nx_G = nx.DiGraph()

    # for source, nbr_weights in node_to_edges.iteritems():
    for source, nbr_weights in node_to_edges.items():
        for nbr_weight in nbr_weights:
            target = nbr_weight[0]

            if opts.trans_type == 0:
                edge_weight = get_edge_weight(source, target) + pseudo_count
                weight = edge_weight

            elif opts.trans_type == 1:
                target_nbrs = node_to_edges.get(target, None)
                local_degree = 0 if target_nbrs is None else len(target_nbrs)
                local_degree += pseudo_count
                weight = local_degree

            else:
                global_degree = nbr_weight[1] + pseudo_count
                weight = global_degree

            nx_G.add_edge(source, target, weight=weight)

    # List of the starting nodes.
    roots = list()
    # List of the starting nodes excluding nodes without outgoing neighbors.
    roots_noleaf = list()

    str_list = list()
    str_list.append(parts[0])

    probs = list()
    probs_noleaf = list()
    weight_sum_noleaf = 0.0
    weight_sum = 0.0

    # Obtain sampling probabilities of roots.
    # for node, weight in nx_G.out_degree_iter(weight="weight"):
    for node, weight in nx_G.out_degree(weight="weight"):
        org_weight = weight
        if weight == 0:
            weight += pseudo_count
        weight_sum += weight
        if org_weight > 0:
            weight_sum_noleaf += weight

    for node, weight in nx_G.out_degree(weight="weight"):
        org_weight = weight
        if weight == 0:
            weight += pseudo_count
        roots.append(node)
        prob = weight / weight_sum
        probs.append(prob)
        if org_weight > 0:
            roots_noleaf.append(node)
            prob = weight / weight_sum_noleaf
            probs_noleaf.append(prob)

    sample_total = opts.walks_per_graph
    first_time = True
    G = node2vec.Graph(nx_G, True, opts.p, opts.q)
    G.preprocess_transition_probs()

    while True:
        if first_time:
            first_time = False
            node_list = roots
            prob_list = probs
        else:
            node_list = roots_noleaf
            prob_list = probs_noleaf
        n_sample = min(len(node_list), sample_total)
        if n_sample <= 0:
            break
        sample_total -= n_sample

        sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)
        walks = G.simulate_walks(len(sampled_nodes), opts.walk_length, sampled_nodes)
        for walk in walks:
            str_list.append(' '.join(str(k) for k in walk))
    return '\t'.join(str_list), label, observation_num


def file_len(fname):
    lines = 0
    for line in open(fname):
        lines += 1
    return lines


def read_graphs(which_set):
    graph_cnt = 0
    graph_file = cascade_file_prefix + which_set + ".txt"
    graph_walk_file = graph_walk_prefix + which_set + ".txt"
    num_graphs = file_len(graph_file)
    write_file = open(graph_walk_file, 'w')
    rfile = open(graph_file, 'r')
    start_time = time.time()
    mid_label = {}
    mid_observation_num = {}
    num = 0
    for line in rfile:
        line = line.rstrip('\r\n')
        mid = line.split('\t')[0]
        walk_string, label, observation_num = parse_graph(line)
        if walk_string:  # judge, the walk string can be None my add
            write_file.write(walk_string + "\n")
            mid_label[mid] = label
            mid_observation_num[mid] = observation_num
            num += 1
        graph_cnt += 1
        # if graph_cnt > opts.sample_num:  # control the total samples
        #   break
        if graph_cnt % 1000 == 0:
            print("Processed graphs in %s set: %d/%d" % (which_set, graph_cnt, num_graphs))
            # print(mid_label)
            # exit(0)
    print(which_set, ":", num)

    # save mid:label in this way
    pickle.dump(mid_label, open(data_path + 'label_{}.pkl'.format(which_set), 'wb'))
    # save mid:observation_num in this way
    pickle.dump(mid_observation_num, open(data_path + 'observation_num_{}.pkl'.format(which_set), 'wb'))

    print("--- %.2f seconds per graphs in %s set ---" % ((time.time() - start_time) / graph_cnt, which_set))
    rfile.close()
    write_file.close()


def read_walks_set(which_set, walks):
    graph_walk_file = graph_walk_prefix + which_set + ".txt"
    rfile = open(graph_walk_file, 'r')
    for line in rfile:
        line = line.rstrip('\r\n')
        walk_strings = line.split('\t')
        for i, walk_str in enumerate(walk_strings):
            if (i == 0):
                continue
            walks.append(walk_str.split(" "))
    rfile.close()


def learn_embeddings(walks, embeding_size):
    embed_file = embed_prefix + str(embeding_size) + ".txt"
    print(embed_file)
    # Learn embeddings by optimizing the Skipgram objective using SGD.
    #
    # old version of Word2Vec
    '''
    model = Word2Vec(walks, size=embeding_size, window=opts.window_size, min_count=0, sg=1, workers=opts.workers, iter=opts.iter)
    '''
    try:
        model = Word2Vec(walks, vector_size=embeding_size, window=opts.window_size, min_count=0, sg=1, workers=opts.workers, epochs=opts.emb_epoch)
    except TypeError as e:
        print("use old version of Word2Vec")
        model = Word2Vec(walks, size=embeding_size, window=opts.window_size, min_count=0, sg=1, workers=opts.workers, iter=opts.iter)

    # model.save_word2vec_format(embed_file)
    model.wv.save_word2vec_format(embed_file)


if __name__ == "__main__":

    # save node degree here

    # get_global_info()

    # new global info is more convenient 修改部分
    get_global_info_new()

    # import pickle

    # exit(0)

    start = time.time()
    for which_set in sets:
        read_graphs(which_set)
    print(time.time() - start)
    opts.workers = 10
    print("Train word2vec with dimension " + str(opts.dimensions))
    start = time.time()
    walks = list()
    read_walks_set(sets[0], walks)
    print('consuming {} seconds to generate walks'.format(time.time() - start))

    # the embedding is kind of useless
    learn_embeddings(walks, opts.dimensions)

    # add save node_id degree

    print(time.time() - start)
