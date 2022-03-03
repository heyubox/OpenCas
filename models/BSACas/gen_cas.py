from sklearn.utils import shuffle
from config import opts
import os
from utils.process_cascade import parse_line, get_observation_path, parse_sequence_counts, parse_observation_path, qualified_cascade
import numpy as np
import pickle
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import math

sys.path.append('./')

num_try = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(num_try)


def load_observation_paths(file_name, qualified_data_ids, observation_time, load_path=False):
    cascadeID_list = []
    if load_path and os.path.exists((opts.DATA_PATH + 'observation_paths_{}.pkl').format(observation_time)):
        observation_paths_total = pickle.load(open((opts.DATA_PATH + 'observation_paths_{}.pkl').format(observation_time), 'rb'))
        labels = pickle.load(open((opts.DATA_PATH + 'labels_{}.pkl').format(observation_time), 'rb'))

    else:
        observation_paths_total = []
        labels = []
        num_lines = sum(1 for line in open(file_name, 'r'))

        for line in tqdm(open(file_name), total=num_lines):

            parts = line.split('\t')
            cascade_id, _, _, _, paths = parse_line(parts)
            if cascade_id not in qualified_data_ids:
                continue
            observation_paths, label = get_observation_path(paths, observation_time, return_label=True)
            labels.append(label)
            observation_paths_total.append(observation_paths)
            cascadeID_list.append(int(cascade_id))


        pickle.dump(observation_paths_total, open((opts.DATA_PATH + 'observation_paths_{}.pkl').format(observation_time), 'wb'))
        pickle.dump(labels, open((opts.DATA_PATH + 'labels_{}.pkl').format(observation_time), 'wb'))
    if not load_path:
        return observation_paths_total, labels, cascadeID_list
    else:
        return observation_paths_total, labels


def get_x_y(file_name, observation_time, qualified_data_ids, interval=180, load_path=False, load_data=False, input_features=100):

    chunk = 0
    if load_data and os.path.exists((opts.DATA_PATH + 'X_{}_{}_{}_{}_new.pkl').format(observation_time, True, interval, chunk)):
        X = pickle.load(open((opts.DATA_PATH + 'X_{}_{}_{}_{}_new.pkl').format(observation_time, True, interval, chunk)))
        Y = []
        for g_sequence in X:
            g = g_sequence[-1]
            Y.append(g.label)
        return X, Y

    X = []
    Y = []
    qualified_user_ids = []
    observation_paths_total, labels, cascadeID_list = load_observation_paths(file_name, qualified_data_ids, observation_time, load_path=load_path)

    if load_data and os.path.exists((opts.DATA_PATH + 'interval_indexes_{}_{}.pkl').format(observation_time, interval)):
        interval_indexes = pickle.load(open((opts.DATA_PATH + 'interval_indexes_{}_{}.pkl').format(observation_time, interval), 'rb'))
    else:
        interval_indexes = []
        for observation_paths in observation_paths_total:
            for p in observation_paths:
                qualified_user_ids+=p[0]
                # qualified_user_ids.add(p[0][-1])
            interval_index = parse_sequence_counts(observation_paths, interval, observation_time)
            interval_indexes.append(interval_index)
        pickle.dump(interval_indexes, open((opts.DATA_PATH + 'interval_indexes_{}_{}.pkl').format(observation_time, interval), 'wb'))
    qualified_user_ids = set(qualified_user_ids)
    idmap = dict(zip(qualified_user_ids,range(len(qualified_user_ids))))
    pickle.dump(idmap,open((opts.DATA_PATH + 'idmap_{}.pkl').format(observation_time), 'wb'))

    num = 0
    chunk = 0
    for (observation_paths, label, interval_index) in tqdm(zip(observation_paths_total, labels, interval_indexes)):
        num += 1

        if num == 100000:
            print("\n==========sorry too many cascade=======\n")
            # break

        interval_indexes.append(interval_index)

        g_sequence = []
        for i, index in enumerate(interval_index):
            g = parse_observation_path(observation_paths[:int(index)], label, directed=False, feature_len=input_features)
            map = {}
            for key,value in zip(g.id2nodes.keys(),g.id2nodes.values()):
                map[key]=idmap[value]
            g.node2idmap = map
            g_sequence.append(g)


        if len(g_sequence) <= 2:
            continue

        X.extend(g_sequence)

        Y.append(label)

    pickle.dump(X, open((opts.DATA_PATH + 'X_{}_{}_{}_{}_new.pkl').format(observation_time, True, interval, chunk), 'wb'))
    chunk += 1

    return X, Y, cascadeID_list


if __name__ == '__main__':

    observation_time = opts.observation_time
    interval = opts.interval
    # interval = 120
    file_name = opts.RAWDATA_PATH
    # n_embs = 1000

    input_features = opts.input_features
    n_seq = interval_num = math.ceil(observation_time / interval)
    fliter_threshold = 100000
    load = False

    start = time.time()

    load_id = load
    if load_id and os.path.exists((opts.DATA_PATH + 'qualified_data_ids_{}.pkl').format(observation_time)):
        qualified_data_ids = pickle.load(open((opts.DATA_PATH + 'qualified_data_ids_{}.pkl').format(observation_time), 'rb'))
    else:
        qualified_data_ids = qualified_cascade(file_name, observation_time, pred_time=opts.prediction_time)
        pickle.dump(qualified_data_ids, open((opts.DATA_PATH + 'qualified_data_ids_{}.pkl').format(observation_time), 'wb'))

    X, Y, cascadeID_list = get_x_y(file_name, observation_time, qualified_data_ids, interval, load_path=load, load_data=load, input_features=input_features)
    print("==============generation complete===============")
    print("caslist", len(cascadeID_list))
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape([-1, interval_num])
    if_shuffle = True
    seed = opts.random_seed
    # Use 15% for test, 15% for valid, and 70% for training. 随机选择
    train_id, test_id, _, _ = train_test_split(cascadeID_list, range(len(cascadeID_list)), test_size=1 - 0.7, random_state=seed, shuffle=if_shuffle)
    test_id, val_id, _, _ = train_test_split(test_id, range(len(test_id)), test_size=0.5, random_state=seed, shuffle=if_shuffle)
    np.savetxt(opts.DATA_PATH + '10-10-10_train_test_val.txt', (train_id[:10], test_id[:10], val_id[:10]))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - 0.7, random_state=seed, shuffle=if_shuffle)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, Y_test, test_size=0.5, random_state=seed, shuffle=if_shuffle)

    # x_train_index = pickle.load(open('index_train.pkl', 'rb'))
    # x_test_index = pickle.load(open('index_test.pkl', 'rb'))
    # x_valid_index = pickle.load(open('index_valid.pkl', 'rb'))
    #
    # X_train, X_test, X_valid, Y_train, Y_test, Y_valid = \
    #     X[x_train_index], X[x_test_index], X[x_valid_index],\
    #     Y[x_train_index], Y[x_test_index], Y[x_valid_index]

    X_train = X_train[Y_train <= fliter_threshold]
    X_test = X_test[Y_test <= fliter_threshold]
    X_valid = X_valid[Y_valid <= fliter_threshold]
    Y_train = Y_train[Y_train <= fliter_threshold]
    Y_test = Y_test[Y_test <= fliter_threshold]
    Y_valid = Y_valid[Y_valid <= fliter_threshold]

    Y_train, Y_valid, Y_test = np.log2(Y_train + 1.0), np.log2(Y_valid + 1.0), np.log2(Y_test + 1.0)

    pickle.dump(X_train, open(opts.DATA_PATH + "train.pkl", 'wb'))
    pickle.dump(Y_train, open(opts.DATA_PATH + "train_labels.pkl", 'wb'))

    pickle.dump(X_valid, open(opts.DATA_PATH + "val.pkl", 'wb'))
    pickle.dump(Y_valid, open(opts.DATA_PATH + "val_labels.pkl", 'wb'))

    pickle.dump(X_test, open(opts.DATA_PATH + "test.pkl", 'wb'))
    pickle.dump(Y_test, open(opts.DATA_PATH + "test_labels.pkl", 'wb'))

    print("len(Y_train):{}\n len(Y_valid):{}\n len(Y_test):{}\n".format(len(Y_train), len(Y_valid), len(Y_test)))
