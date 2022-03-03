from config import opts
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import os
import pickle
'''
format FROM(DeepHawkes):
<message_id>\tab<user_id>\tab<publish_time>\tab<retweet_number>\tab<retweets>
<message_id>:     the unique id of each message, ranging from 1 to 119,313.
<root_user_id>:   the unique id of root user. The user id ranges from 1 to 6,738,040.
<publish_time>:   the publish time of this message, recorded as unix timestamp.
<retweet_number>: the total number of retweets of this message within 24 hours.
<retweets>:       the retweets of this message, each retweet is split by " ". Within each retweet, it records 
the entile path for this retweet, the format of which is <user1>/<user2>/......<user n>:<retweet_time>.

TO(DeepCas):
`cascade_id `\t `starter_id`... \t `constant_field `\t `num_nodes `\t `source:target:timestamp`... \t `labels`...
'''


def parse_line(parts):
    cascade_id = parts[0]
    # msg_time = int(parts[2])
    msg_time = parts[2]
    try:
        hour = time.strftime("%H", time.localtime(msg_time))
    except:
        try:
            ts = time.strptime(parts[2], '%Y-%m-%dT%H:%M:%S')
            hour = ts.tm_hour
        except:
            ts = time.localtime(int(parts[2]))
            hour = ts.tm_hour

    hour = int(hour)
    retweet_number = int(parts[3])
    paths = parts[4].split(' ')
    return cascade_id, msg_time, hour, retweet_number, paths


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


def gen_cascade_graph_weibo(observation_time, observation_num, pre_times, filename, load_valid=None):
    '''
    @ FORMAT: `cascade_id `\\t`starter_id`...\\t`constant_field `\\t`num_nodes `\\t`source:target:timestamp`...\\t`labels`...
    '''
    if load_valid is not None:
        vid = pickle.load(open(load_valid, 'rb'))
        vid = vid.keys()
    else:
        vid = None
    file = open(filename)
    cascades_total = dict()
    global_graph = []
    res = []
    discard_midnight = 0
    discard_outer = 0
    num_lines = sum(1 for line in open(filename, 'r'))
    for line in tqdm(file, total=num_lines):
        parts = line.strip().split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        if load_valid is not None:
            if cascadeID not in vid:
                continue
        # print(cascadeID)
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path):
            print('wrong number of nodes', n_nodes, len(path))
        hour = get_hour(parts[2], filename)
        # to keep the same with
        if hour <= opts.start_hour or hour >= opts.end_hour:  # 8-18
            discard_midnight += 1
            continue

        starter_id = []
        observation_path = []
        labels = []
        edges = set()
        in_ob = 0
        for p in path:
            nodes = p.split(":")[0].split("/")
            if len(nodes) == 1:
                nodes.append(nodes[0])
            nodes_ok = True
            for n in nodes:
                if int(n) == -1:
                    nodes_ok = False
            if not (nodes_ok):
                print("error id at cas_id {}".format(cascadeID))
                print(nodes)
                continue
            # print nodes

            time_now = int(p.split(":")[1])

            # 读取全图global graph/部分图local graph和全cascade
            # 后续会筛选cascade但是不会筛选图
            for i in range(1, len(nodes)):
                edges.add(nodes[i - 1] + ":" + nodes[i] + ":{}".format(time_now))
                if time_now < observation_time:  # 加上限制条件后只读取local graph
                    #         edges.add(nodes[i - 1] + ":" + nodes[i] + ":{}".format(time_now))
                    global_graph.append(nodes[i - 1] + " " + nodes[i])

            if opts.time_or_number == 'Time':
                # if time_now < observation_time:# gen_walks 会再根据op计算label，此处全保留
                starter = nodes[-2]
                # follower = nodes[-1]
                starter_id.append(starter)
                observation_path.append(",".join(nodes) + ":" + str(time_now))
                if time_now < observation_time:
                    in_ob += 1

            else:
                # my addition by observation_num
                print("unsurport Nums")
                return None
                # if len(observation_path) <= observation_num+1 and time_now < pre_times[0]:
                #     observation_path.append(",".join(nodes) + ":" + str(time_now))
                #     for i in range(1, len(nodes)):
                #         edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

        if in_ob < opts.least_num or in_ob > opts.up_num:
            # print(cascadeID)
            discard_outer += 1
            # least_rewteetnum = config.least_num   # 5 or 10
            # if len(observation_path) < least_rewteetnum:
            continue
        # try:
        #     cascades_total[cascadeID] = msg_time
        # except:
        for i in range(len(pre_times)):
            labels.append(len(path))

        # 读取全图global graph/部分图local graph和全cascade
        # 后续会筛选cascade但是不会筛选图
        # for p in path:
        #     nodes = p.split(":")[0].split("/")
        #     time_now = int(p.split(":")[1])
        #     for i in range(1, len(nodes)):
        #         edges.add(nodes[i - 1] + "/" + nodes[i] +
        #                   ":{}".format(time_now))

        cascades_total[cascadeID] = hour

        complete_cas = "{cascade_id}\t{starter_id}\t{constant_field}\t{num_nodes}\t{source_target_weight}\t{label}" \
        .format(cascade_id=cascadeID,starter_id=starter_id[0],constant_field=parts[2],num_nodes=len(observation_path),source_target_weight=' '.join(edges),label=' '.join(str(e) for e in labels))

        res.append(complete_cas)

    n_total = len(cascades_total)
    print("total_readin:", num_lines)
    print("discard_midnight:", discard_midnight)
    print("discard_outer:", discard_outer)
    print('total:', n_total)

    file.close()
    return res, set(global_graph)


def data_to_cascadeFile(fname, data):
    with open(fname, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


def data_to_graphFile(fname, data):
    with open(fname, 'w') as f:
        for index, item in tqdm(enumerate(data)):
            if not item:
                print("empty graph")
                pass
            else:
                # f.write("{}\t".format(str(index)))
                # for k,v in item.items():
                f.write(item + '\n')


def split_dataset(data_set, graph_global, train_per, test_per, val_per, filename_train, filename_val, filename_test, filename_graph, if_shuffle=True):
    assert (train_per + test_per + val_per) == 1
    from sklearn.model_selection import train_test_split
    print("dataset_len:", str(len(data_set)))
    X_train, X_test, y_train, y_test = train_test_split(data_set, [None] * len(data_set), test_size=1 - train_per, random_state=opts.random_seed, shuffle=if_shuffle)

    val_per = val_per / (test_per + val_per)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_per, random_state=opts.random_seed, shuffle=if_shuffle)
    print('train data:', len(X_train))
    print('valid data:', len(X_val))
    print('test data:', len(X_test))

    data_to_cascadeFile(filename_train, X_train)
    data_to_cascadeFile(filename_test, X_test)
    data_to_cascadeFile(filename_val, X_val)
    data_to_graphFile(filename_graph, graph_global)


if __name__ == "__main__":
    filename = opts.RAWDATA_PATH
    filename_train = opts.cascade_train
    filename_test = opts.cascade_test
    filename_val = opts.cascade_val
    filename_graph = opts.global_graph_file
    if opts.is_weibo:
        res, g_b = gen_cascade_graph_weibo(opts.observation_time, opts.observation_num, [opts.prediction_time], filename)
        # res,g_b = gen_cascade_graph_weibo(opts.observation_time,opts.observation_num,[opts.prediction_time],filename,load_valid='../valid_id_100_train_test_valid.pkl')
    else:
        exit(0)

    split_dataset(res, g_b, 0.7, 0.15, 0.15, filename_train, filename_val, filename_test, filename_graph)
