import networkx as nx
import time

from scipy.sparse.construct import rand
# from functools import cmp_to_key
from config import opts
import sys
import pickle
import os
import time
from tqdm import tqdm


def get_hour(time_str, filename):
    hour = None
    try:
        msg_time = int(time_str)
        hour = time.strftime("%H", time.localtime(msg_time))
        hour = int(hour)
    except:
        if '170w' in filename:  # fixed in 11.15, however, in this way, more datasets will be removed
            ts = time.strptime(time_str, '%Y-%m-%d-%H:%M:%S')
            hour = ts.tm_hour
        elif 'castle' in filename:
            # for data castle weibo
            hour = int(time_str[:2])
        elif 'smp' in filename:
            ts = time.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            hour = ts.tm_hour
        else:
            print('wrong time format')
    return hour


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


def gen_cascades_obser(observation_time, pre_times, filename):
    cascades_total = dict()
    cascades_type = dict()
    discard_midnight = 0
    discard_outer = 0
    num_lines = sum(1 for line in open(filename, 'r'))

    with open(filename) as f:
        for line in tqdm(f, total=num_lines):
            parts = line.split("\t")
            if len(parts) != 5:
                print('wrong format!')
                continue
            cascadeID = parts[0]
            n_nodes = int(parts[3])
            path = parts[4].split(" ")
            if n_nodes != len(path):
                print('wrong number of nodes', n_nodes, len(path))
            msg_pub_time = parts[2]

            observation_path = []
            labels = []
            edges = set()
            for i in range(len(pre_times)):
                labels.append(0)
            for p in path:
                nodes = p.split(":")[0].split("/")
                nodes_ok = True
                for n in nodes:
                    if int(n) == -1:  # delete invalid id
                        nodes_ok = False
                if not (nodes_ok):
                    print("error id at cas_id {}".format(cascadeID))
                    print(nodes)
                    continue
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                for i in range(len(pre_times)):
                    if time_now < pre_times[i]:
                        labels[i] += 1

            # if labels[0] > 1000:  # most are noisy or wrong dataset
            #     continue
            # if labels[0] > opts.up_num:  # most are noisy or wrong dataset
            #     continue

            if len(observation_path) < opts.least_num or len(observation_path) > opts.up_num:
                discard_outer += 1

                continue

            # for 170weibo, we use hour instead of msg_pub_time
            # try:
            #     cascades_total[cascadeID] = int(msg_pub_time)
            # except:
            try:
                ts = time.strptime(parts[2], '%Y-%m-%d-%H:%M:%S')
                hour = ts.tm_hour
            except:
                msg_time = time.localtime(int(parts[2]))
                hour = time.strftime("%H", msg_time)
                hour = int(hour)

            if hour <= opts.start_hour or hour >= opts.end_hour:
                discard_midnight += 1

                continue

            cascades_total[cascadeID] = hour

        n_total = len(cascades_total)
        print("total_readin:", num_lines)
        print("discard_midnight:", discard_midnight)
        print("discard_outer:", discard_outer)
        print('total:', n_total)
        # key = cmp_to_key(lambda x, y: int(x[1]) - int(y[1]))
        # sorted_msg_time = sorted(cascades_total.items(), key=key)
        import operator
        sorted_msg_time = sorted(cascades_total.items(), key=operator.itemgetter(1))

        cascades_total_key = list(cascades_total.keys())
        seed = opts.random_seed
        if_shuffle = True
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(cascades_total_key, [None] * len(cascades_total_key), test_size=1 - 0.7, random_state=seed, shuffle=if_shuffle)

        val_per = 0.15 / (0.15 + 0.15)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_per, random_state=seed, shuffle=if_shuffle)
        for k in X_train:
            cascades_type[k] = 1
        for k in X_val:
            cascades_type[k] = 2
        for k in X_test:
            cascades_type[k] = 3

        print('train data:', len([cid for cid in cascades_type if cascades_type[cid] == 1]))
        print('valid data:', len([cid for cid in cascades_type if cascades_type[cid] == 2]))
        print('test data:', len([cid for cid in cascades_type if cascades_type[cid] == 3]))

    return cascades_total, cascades_type


def gen_cascades_citation_obser(observation_time, pre_times, filename):
    cascades_total = dict()
    cascades_type = dict()
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) != 5:
                print('wrong format!')
                continue
            cascadeID = parts[0]
            n_nodes = int(parts[3])
            path = parts[4].split(" ")
            if n_nodes != len(path) and n_nodes != len(path) - 1:
                print('wrong number of nodes', n_nodes, len(path))
            msg_pub_time = parts[2]

            observation_path = []
            labels = []
            edges = set()
            for i in range(len(pre_times)):
                labels.append(0)
            for p in path:
                nodes = p.split(":")[0].split("/")
                nodes_ok = True
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                for i in range(len(pre_times)):
                    if time_now < pre_times[i]:
                        labels[i] += 1

            # for citation, we use month instead of msg_pub_time
            ts = time.strptime(parts[2], '%Y-%m-%d')
            hour = ts.tm_mon

            if len(observation_path) < opts.least_num:
                continue

            cascades_total[cascadeID] = hour

        n_total = len(cascades_total)
        print('total:', n_total)

        # key = cmp_to_key(lambda x, y: int(x[1]) - int(y[1]))
        # sorted_msg_time = sorted(cascades_total.items(), key=key)
        import operator
        sorted_msg_time = sorted(cascades_total.items(), key=operator.itemgetter(1))

        count = 0
        for (k, v) in sorted_msg_time:
            if count < n_total * 1.0 / 20 * 14:
                cascades_type[k] = 1
            elif count < n_total * 1.0 / 20 * 17:
                cascades_type[k] = 2
            else:
                cascades_type[k] = 3
            count += 1

        print('train data:', len([cid for cid in cascades_type if cascades_type[cid] == 1]))
        print('valid data:', len([cid for cid in cascades_type if cascades_type[cid] == 2]))
        print('test data:', len([cid for cid in cascades_type if cascades_type[cid] == 3]))

    return cascades_total, cascades_type


def discard_cascade(observation_time, pre_times, filename):
    discard_cascade_id = dict()
    num_cas = 0
    num_lines = sum(1 for line in open(filename, 'r'))
    with open(filename) as f:
        for line in tqdm(f, total=num_lines):
            num_cas += 1
            # if (num_cas + 1) % 100==0:
            #     print('\rprocessed {}'.format(num_cas))
            parts = line.split("\t")
            if len(parts) != 5:
                print('wrong format!')
                continue
            cascadeID = parts[0]
            n_nodes = int(parts[3])
            path = parts[4].split(" ")
            if n_nodes != len(path):
                print('wrong number of nodes', n_nodes, len(path))
            hour = get_hour(parts[2], filename)
            if hour <= opts.start_hour or hour >= opts.end_hour:  # 8-18
                continue
            observation_path = []
            edges = set()
            for p in path:
                nodes = p.split(":")[0].split("/")
                nodes_ok = True
                for n in nodes:
                    if int(n) == -1:  # delete invalid id
                        nodes_ok = False
                if not (nodes_ok):
                    print("error id at cas_id {}".format(cascadeID))
                    print(nodes)
                    continue
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            nx_Cass = nx.DiGraph()
            for i in edges:
                part = i.split(":")
                source = part[0]
                target = part[1]
                weight = part[2]
                nx_Cass.add_edge(source, target, weight=weight)

            # -------------------to speed up
            num = nx_Cass.number_of_nodes()
            # if num>100: # 过滤掉大于N的数据
            if num > opts.up_num or num < opts.least_num:
                if cascadeID not in discard_cascade_id:  # 1为丢弃
                    discard_cascade_id[cascadeID] = 1
            else:
                discard_cascade_id[cascadeID] = 0

            try:
                L = directed_laplacian_matrix(nx_Cass)
            except:
                discard_cascade_id[cascadeID] = 1
                s = sys.exc_info()

    return discard_cascade_id


def directed_laplacian_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95):
    import numpy as np
    import scipy as sp
    from scipy.sparse import identity, spdiags, linalg
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = "random"
            else:
                walk_type = "lazy"
        else:
            walk_type = "pagerank"

    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, dtype=float)
    n, m = M.shape
    if walk_type in ["random", "lazy"]:
        DI = spdiags(1.0 / np.array(M.sum(axis=1).flat), [0], n, n)
        if walk_type == "random":
            P = DI * M
        else:
            I = identity(n)
            P = (I + DI * M) / 2.0

    elif walk_type == "pagerank":
        if not (0 < alpha < 1):
            raise nx.NetworkXError('alpha must be between 0 and 1')
        M = M.todense()
        dangling = np.where(M.sum(axis=1) == 0)
        for d in dangling[0]:
            M[d] = 1.0 / n
        M = M / M.sum(axis=1)
        P = alpha * M + (1 - alpha) / n
    else:
        raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")

    evals, evecs = linalg.eigs(P.T, k=1, tol=1E-2)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = np.lib.scimath.sqrt(p)
    Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
    I = np.identity(len(G))
    return I - (Q + Q.T) / 2.0


def gen_cascade(observation_time, pre_times, filename, filename_ctrain, filename_cval, filename_ctest, filename_strain, filename_sval, filename_stest, cascades_type, discard_cascade_id):
    file = open(filename, "r")
    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")
    num = 0

    num_train, num_valid, num_test = 0, 0, 0
    train_ids, valid_ids, test_ids = [], [], []

    for line in file:
        parts = line.split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path):
            print('wrong number of nodes', n_nodes, len(path))
        try:
            msg_time = time.localtime(int(parts[2]))
            hour = time.strftime("%H", msg_time)
            hour = int(hour)
        except:
            try:
                ts = time.strptime(parts[2], '%Y-%m-%d-%H:%M:%S')
                hour = ts.tm_hour
            except:
                try:
                    ts = time.strptime(parts[2], '%Y-%m-%dT%H:%M:%S')
                    hour = ts.tm_hour
                except:
                    msg_time = time.localtime(int(parts[2]))
                    hour = time.strftime("%H", msg_time)
                    hour = int(hour)

        if hour <= opts.start_hour or hour >= opts.end_hour:
            continue

        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            time_now = int(p.split(":")[1])
            if time_now < observation_time:
                observation_path.append(",".join(nodes) + ":" + str(time_now))
                for i in range(1, len(nodes)):
                    if (nodes[i - 1] + ":" + nodes[i] + ":" + str(time_now)) in edges:
                        continue
                    else:
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":" + str(time_now))
            for i in range(len(pre_times)):
                if time_now < pre_times[i]:
                    labels[i] += 1

        # if len(observation_path) < opts.least_num:
        #     continue

        if len(observation_path) < opts.least_num and len(observation_path) > opts.up_num:
            continue

        for i in range(len(labels)):
            labels[i] = str(labels[i] - len(observation_path))
        if len(edges) <= 1:
            continue

        # 原来判断是否丢掉一部分（>100的数据），这里全部保留
        # if cascadeID in cascades_type and cascades_type[cascadeID] == 1:
        #     file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")#shortespath_train
        #     file_ctrain.write(cascadeID+"\t"+parts[1]+"\t"+parts[2]+"\t"+str(len(observation_path))+"\t"+" ".join(edges)+"\t"+" ".join(labels)+"\n")#cascade_train part[1]-user_id parts[2]-publis_time observation_path "".join(edges) "".join(labels)
        #     num_train += 1
        #     train_ids.append(cascadeID)
        # elif cascadeID in cascades_type and cascades_type[cascadeID] == 2:
        #     file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
        #     file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
        #     num_valid += 1
        #     valid_ids.append(cascadeID)
        # elif cascadeID in cascades_type and cascades_type[cascadeID] == 3 :
        #     file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
        #     file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
        #     num_test += 1
        #     test_ids.append(cascadeID)

        if cascadeID in cascades_type and cascades_type[cascadeID] == 1 and discard_cascade_id[cascadeID] == 0:
            num_train += 1
            file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")  # shortespath_train
            file_ctrain.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) +
                              "\n")  # cascade_train part[1]-user_id parts[2]-publis_time observation_path "".join(edges) "".join(labels)
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2 and discard_cascade_id[cascadeID] == 0:
            num_valid += 1
            file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3 and discard_cascade_id[cascadeID] == 0:
            num_test += 1
            file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")

        num += 1
    print('total qualified cascade', num)

    print("train:, valid:, test:,", num_train, num_valid, num_test)

    import pickle
    dataset_name = opts.dataset
    pickle.dump(train_ids, open(opts.DATA_PATH + 'train_{}_ids.pkl'.format(observation_time), 'wb'))
    pickle.dump(valid_ids, open(opts.DATA_PATH + 'valid_{}_ids.pkl'.format(observation_time), 'wb'))
    pickle.dump(test_ids, open(opts.DATA_PATH + 'test_{}_ids.pkl'.format(observation_time), 'wb'))

    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()


def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print("length of original isd:", len(original_ids))
    return original_ids


def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = []  # walk[0] = cascadeID
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0]  # node
                t = walks[i].split(":")[1]  # time
                graphs[walks[0]].append([[int(xx) for xx in s.split(",")], int(t)])
    return graphs


if __name__ == "__main__":
    observation_time = opts.observation_time
    pre_times = [opts.prediction_time]
    is_weibo = opts.is_weibo
    import time
    start = time.time()
    if is_weibo:
        cascades_total, cascades_type = gen_cascades_obser(observation_time, pre_times, opts.RAWDATA_PATH)
    else:
        cascades_total, cascades_type = gen_cascades_citation_obser(observation_time, pre_times, opts.RAWDATA_PATH)

    print('finish gen cascades obser')
    print(time.time() - start)
    load_discard = False
    if load_discard and os.path.exists(opts.DATA_PATH + 'discard_cascade_id.pkl'):
        discard_cascade_id = pickle.load(open(opts.DATA_PATH + 'discard_cascade_id.pkl', 'rb'))
    else:
        discard_cascade_id = discard_cascade(observation_time, pre_times, opts.RAWDATA_PATH)
        pickle.dump(discard_cascade_id, open(opts.DATA_PATH + 'discard_cascade_id.pkl', 'wb'))

    print(time.time() - start)
    print('length discard_cascade_id:', len(discard_cascade_id), len([id for id in discard_cascade_id.values() if id == 1]))
    print("generate cascade new!!!")
    # discard_cascade_id = {}  # CasCN丢掉了一些cascade，这里先不考虑

    if is_weibo:
        gen_cascade(observation_time, pre_times, opts.RAWDATA_PATH, opts.cascade_train, opts.cascade_val, opts.cascade_test, opts.shortestpath_train, opts.shortestpath_val, opts.shortestpath_test,
                    cascades_type, discard_cascade_id)
    print(time.time() - start)
