import numpy as np
import pickle
import config

opts = config.opts
DATA_PATH = opts.DATA_PATH
tail = opts.tail
LABEL_NUM = 0

import time

start = time.time()

labels_train = {}
sizes_train = {}
with open(DATA_PATH + 'cascade_train%s.txt' % tail, 'r') as f:
    for line in f:
        profile = line.split('\t')
        labels_train[profile[0]] = profile[-1]
        sizes_train[profile[0]] = int(profile[3])

graphs_train = {}
with open(DATA_PATH + 'random_walks_train%s.txt' % tail, 'r') as f:
    for line in f:
        walks = line.strip().split('\t')
        graphs_train[walks[0]] = []
        for i in range(1, len(walks)):
            graphs_train[walks[0]].append([int(x) for x in walks[i].split()])

graphs_val = {}
with open(DATA_PATH + 'random_walks_val%s.txt' % tail, 'r') as f:
    for line in f:
        walks = line.strip().split('\t')
        graphs_val[walks[0]] = []
        for i in range(1, len(walks)):
            graphs_val[walks[0]].append([int(x) for x in walks[i].split()])

graphs_test = {}
with open(DATA_PATH + 'random_walks_test%s.txt' % tail, 'r') as f:
    for line in f:
        walks = line.strip().split('\t')
        graphs_test[walks[0]] = []
        for i in range(1, len(walks)):
            graphs_test[walks[0]].append([int(x) for x in walks[i].split()])

sets = ['train_new', 'val_new', 'test_new']
labels_train = pickle.load(open(DATA_PATH + 'label_{}.pkl'.format(sets[0]), 'rb'))
observation_num_train = pickle.load(open(DATA_PATH + 'observation_num_{}.pkl'.format(sets[0]), 'rb'))

labels_val = {}
sizes_val = {}
with open(DATA_PATH + 'cascade_val%s.txt' % tail, 'r') as f:
    for line in f:
        profile = line.split('\t')
        labels_val[profile[0]] = profile[-1]
        sizes_val[profile[0]] = int(profile[3])

labels_val = pickle.load(open(DATA_PATH + 'label_{}.pkl'.format(sets[1]), 'rb'))
observation_num_val = pickle.load(open(DATA_PATH + 'observation_num_{}.pkl'.format(sets[1]), 'rb'))

labels_test = {}
sizes_test = {}
with open(DATA_PATH + 'cascade_test%s.txt' % tail, 'r') as f:
    for line in f:
        profile = line.split('\t')
        labels_test[profile[0]] = profile[-1]
        sizes_test[profile[0]] = int(profile[3])

labels_test = pickle.load(open(DATA_PATH + 'label_{}.pkl'.format(sets[2]), 'rb'))
observation_num_test = pickle.load(open(DATA_PATH + 'observation_num_{}.pkl'.format(sets[2]), 'rb'))


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


original_ids = set()
for key in graphs_train.keys():
    for walk in graphs_train[key]:
        for i in set(walk):
            original_ids.add(i)
for key in graphs_val.keys():
    for walk in graphs_val[key]:
        for i in set(walk):
            original_ids.add(i)
for key in graphs_test.keys():
    for walk in graphs_test[key]:
        for i in set(walk):
            original_ids.add(i)

original_ids.add(-1)
index = IndexDict(original_ids)

# My add add new embedding method: deepwalk/line
# import pickle
# line_emb_file = '/home/tiger/.graphvite/line_weibo_castle.pkl'
# embs = pickle.load(open(line_emb_file,  'rb'))
# node_embs = embs['vertex_embeddings']  # actually each node has two embeddings, we simply use one
# id2node = embs['id2name']
# node2id = dict(zip(id2node, range(len(id2node))))
# num_dims = node_embs.shape[1]
# print('emb_shape', num_dims)
# node_vec = np.random.normal(size=(index.length(), num_dims))
# for node_id in node2id:
#     if int(node_id) not in original_ids:
#         continue
#     node_vec[index.new(int(node_id)), :] = node_embs[node2id[node_id]]

# My add new embedding role2vec
# node2id should be the same with line/deepwalk
# gw_emb_file = '/home/tiger/PycharmProjects/DeepCas/data/gw_castle.weibo.npy'
# # node2id = pickle.load(node2id, open('data/castle_nodes2id.pkl', 'rb'))
# role2vec = np.load(gw_emb_file)
# num_dims = role2vec.shape[1]
# node_vec2 = np.random.normal(size=(index.length(), num_dims))
# for node_id in node2id:
#     if node_id not in original_ids:
#         continue
#     node_vec2[index.new(node_id), :] = role2vec[node2id[node_id]]
#
# node2vec = np.concatenate([node_vec, node_vec2], axis=-1)

# nodes_deg = pickle.load(open(DATA_PATH+"/"+'nodes_deg.pkl', 'rb'))
# np.random.seed(13)

# with open(DATA_PATH+'node_vec_50.txt', 'r') as f:
#     not_in = 0
#     in_num = 0
#     line = f.readline()
#     temp = line.strip().split()
#     num_nodes = int(temp[0])
#     # num_dims = int(temp[1]) + 1  #增加一个degree的维度
#     num_dims = int(temp[1])
#     # node_vec = np.random.normal(size=(index.length(), num_dims))
#     for i in range(num_nodes):
#         line = f.readline()
#         temp = line.strip().split()
#         node_id = int(temp[0])
#         if not node_id in original_ids:
#             continue
#         if str(node_id) not in node2id:
#             not_in += 1
#         else:
#             in_num += 1
#             node_vec[index.new(int(node_id)), :] = node_embs[node2id[str(node_id)]]
#
#         # node_vec[index.new(node_id), :] = np.array([float(temp[j]) for j in range(1, len(temp))])
#         #增加一个degree的维度
#         # node_vec[index.new(node_id), :] = np.array([float(temp[j]) for j in range(1, len(temp))]+[np.log(nodes_deg.get(node_id,0)+1.)])
#
#     print(not_in, in_num)

with open(DATA_PATH + 'node_vec_{}.txt'.format(opts.dimensions), 'r') as f:
    line = f.readline()
    temp = line.strip().split()
    num_nodes = int(temp[0])
    # num_dims = int(temp[1]) + 1  #增加一个degree的维度
    num_dims = int(temp[1])
    node_vec = np.random.normal(size=(index.length(), num_dims))
    for i in range(num_nodes):
        line = f.readline()
        temp = line.strip().split()
        node_id = int(temp[0])
        if not node_id in original_ids:
            continue
        node_vec[index.new(node_id), :] = np.array([float(temp[j]) for j in range(1, len(temp))])

pickle.dump(node_vec, open(DATA_PATH + 'node_vec.pkl', 'wb'), protocol=2)

x_data = []
y_data = []
sz_data = []
for key, graph in graphs_train.items():
    # label = labels_train[key].split()
    # y = int(label[LABEL_NUM])
    y = labels_train[key]
    temp = []
    for walk in graph:
        if len(walk) < 10:
            for i in range(10 - len(walk)):
                walk.append(-1)
        temp.append(index.new(walk))
    x_data.append(temp)
    # y_data.append(np.log(y+1.0)/np.log(2.0))
    y_data.append(np.log2(y + 1.))
    # y_d = float(standard.transform([[np.log(y+1.0)/np.log(2.0)]])[0,0])
    # y_data.append(y_d)
    # sz_data.append(sizes_train[key])  # original code
    # sz_data.append(labels_train[key])
    # sz_data.append(sizes_train[key]-labels_train[key])
    sz_data.append(observation_num_train[key])  # should be this in our dataset

pickle.dump((x_data, y_data, sz_data, index.length()), open(DATA_PATH + 'data_train.pkl', 'wb'), protocol=2)

x_data = []
y_data = []
sz_data = []
for key, graph in graphs_val.items():
    # label = labels_val[key].split()
    # y = int(label[LABEL_NUM])
    y = labels_val[key]
    temp = []
    for walk in graph:
        if len(walk) < 10:
            for i in range(10 - len(walk)):
                walk.append(-1)
        temp.append(index.new(walk))
    x_data.append(temp)
    # y_data.append(np.log(y+1.0)/np.log(2.0))
    y_data.append(np.log2(y + 1.))
    # y_d = float(standard.transform([[np.log(y + 1.0) / np.log(2.0)]])[0, 0])
    # y_data.append(y_data)
    # sz_data.append(labels_val[key])
    # sz_data.append(sizes_val[key])
    # sz_data.append(labels_val[key]-sizes_val[key])
    sz_data.append(observation_num_val[key])

pickle.dump((x_data, y_data, sz_data, index.length()), open(DATA_PATH + 'data_val.pkl', 'wb'), protocol=2)

x_data = []
y_data = []
sz_data = []
for key, graph in graphs_test.items():
    # label = labels_test[key].split()
    # y = int(label[LABEL_NUM])
    y = labels_test[key]
    temp = []
    for walk in graph:
        if len(walk) < 10:
            for i in range(10 - len(walk)):
                walk.append(-1)
        temp.append(index.new(walk))
    x_data.append(temp)
    # y_data.append(np.log(y+1.0)/np.log(2.0))
    y_data.append(np.log2(y + 1.))
    # y_d = float(standard.transform([[np.log(y + 1.0) / np.log(2.0)]])[0, 0])
    # y_data.append(y_d)
    # sz_data.append(sizes_test[key])
    # sz_data.append(labels_test[key])
    # sz_data.append(sizes_test[key]-labels_test[key])
    sz_data.append(observation_num_test[key])

pickle.dump((x_data, y_data, sz_data, index.length()), open(DATA_PATH + 'data_test.pkl', 'wb'), protocol=2)

print(time.time() - start)
