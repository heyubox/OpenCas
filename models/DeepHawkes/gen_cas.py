from tqdm import tqdm
import os
import time
import pickle
from config import opts
import sys

sys.path.append('./')

# print(os.getcwd())


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


def gen_cascade_graph(observation_time, observation_num, pre_times, filename, filename_ctrain, filename_cval, filename_ctest, filename_strain, filename_sval, filename_stest):
    discard_midnight = 0
    discard_outer = 0
    file = open(filename)
    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")
    cascades_total = dict()
    num_lines = sum(1 for line in open(filename, 'r'))
    for line in tqdm(file, total=num_lines):
        parts = line.strip().split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path):
            print(cascadeID,' wrong number of nodes', n_nodes, len(path),path)
            exit(0)

        hour = get_hour(parts[2], filename)
        # to keep the same with
        if hour <= opts.start_hour or hour >= opts.end_hour:  # 8-18
            discard_midnight += 1
            continue

        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        # print(cascadeID)
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
            # print nodes
            time_now = int(p.split(":")[1])

            if opts.time_or_number == 'Time':
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            else:  # my addition by observation_num
                print("unsurport Nums")
                return None
                # if len(observation_path) <= observation_num+1 and time_now < pre_times[0]:
                #     observation_path.append(
                #         ",".join(nodes) + ":" + str(time_now))
                #     for i in range(1, len(nodes)):
                #         edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

        # if labels[0]>1000:
        #     continue

        if len(observation_path) < opts.least_num or len(observation_path) > opts.up_num:

            discard_outer += 1
            continue
        # try:
        #     cascades_total[cascadeID] = msg_time
        # except:
        cascades_total[cascadeID] = hour

    n_total = len(cascades_total)
    print('save valid id in:{}'.format(opts.DATA_PATH))
    pickle.dump(cascades_total, open(opts.DATA_PATH + 'valid_id.pkl', 'wb'))
    print("total_readin:", num_lines)
    print("discard_midnight:", discard_midnight)
    print("discard_outer:", discard_outer)
    print('total:', n_total)
    # import operator
    # sorted_msg_time = sorted(cascades_total.items(),
    #                          key=operator.itemgetter(1))
    cascades_type = dict()
    count = 0
    total_valid_ids = list(cascades_total.keys())
    # 划分数据集
    # 75 15 15
    # 得到valid id: 可选进行shuffle 但是需要randomseed 一致
    if_shuffle = True
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(total_valid_ids, [None] * len(total_valid_ids), test_size=1 - 0.7, random_state=opts.random_seed, shuffle=if_shuffle)

    val_per = 0.15 / (0.15 + 0.15)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_per, random_state=opts.random_seed, shuffle=if_shuffle)
    for k in X_train:
        cascades_type[k] = 1
    for k in X_val:
        cascades_type[k] = 2
    for k in X_test:
        cascades_type[k] = 3

    print('train data:', len([cid for cid in cascades_type if cascades_type[cid] == 1]))
    print('valid data:', len([cid for cid in cascades_type if cascades_type[cid] == 2]))
    print('test data:', len([cid for cid in cascades_type if cascades_type[cid] == 3]))
    num_train, num_valid, num_test = 0, 0, 0

    # to keep the same with CasCN
    # keptids = pickle.load(open(opts.DATA_PATH+'kept_cascade_id.pkl', 'rb'))

    file.close()
    file = open(filename, "r")
    for line in tqdm(file, total=num_lines):
        parts = line.strip('\n').split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]

        if cascadeID not in cascades_type:
            continue

        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path):  # what hell wrong?
            print(parts[0],' wrong number of nodes', n_nodes, len(path))

        try:
            msg_time = time.localtime(int(parts[2]))
            hour = time.strftime("%H", msg_time)
        except:
            hour = int(parts[2][:2])
        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n) == -1:
                    nodes_ok = False
            if not (nodes_ok):
                print(nodes)
                continue
            time_now = int(p.split(":")[1])
            if opts.time_or_number == 'Time':
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            else:
                if len(observation_path) <= observation_num + 1 and time_now < pre_times[0]:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

            for i in range(len(pre_times)):
                # print time,pre_times[i]
                if time_now < pre_times[i]:
                    labels[i] += 1

        for i in range(len(labels)):
            labels[i] = str(labels[i] - len(observation_path))

        # for viral / unviral
        # if cascadeID in viral_cid:
        #     labels[0] = str(0)
        # else:
        #     labels[0] = str(1)

        # if not cascadeID in keptids:
        #     continue

        hour = int(hour)
        if cascadeID in cascades_type and cascades_type[cascadeID] == 1:
            file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctrain.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
            num_train += 1
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2:
            file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
            num_valid += 1
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3:
            file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
            num_test += 1

    print('train', 'test', 'valid', num_train, num_test, num_valid)
    assert n_total == num_valid + num_train + num_test
    print('total data:', num_valid + num_train + num_test)
    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()



if __name__ == "__main__":
    print('===========DeepHawkes loading cascade===========')
    start = time.time()
    observation_time = opts.observation_time
    observation_num = opts.observation_num
    # pre_times = [24 * 3600]
    pre_times = [opts.prediction_time]

    weibo = opts.is_weibo

    gen_cascade_graph(observation_time, observation_num, pre_times, opts.RAWDATA_PATH, opts.cascade_train, opts.cascade_val, opts.cascade_test, opts.shortestpath_train, opts.shortestpath_val,opts.shortestpath_test)


    print('total time in preprocessing:', time.time() - start)
