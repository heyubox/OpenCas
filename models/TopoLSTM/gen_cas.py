import random
import time

import networkx as nx
from config import opts


def generate_cascades(ob_time, pred_time, filename, file_train, file_val, file_test, graph_save_path):

    # a list to save the cascades
    filtered_data = list()
    cascades_type = dict()  # 0 for train, 1 for val, 2 for test
    cascades_time_dict = dict()
    cascades_total = 0
    cascades_valid_total = 0
    discard_midnight = 0
    discard_outer = 0
    with open(filename) as file:
        for line in file:
            # split the cascades into 5 parts
            # 1: cascade id
            # 2: user/item id
            # 3: publish date/time
            # 4: number of adoptions
            # 5: a list of adoptions
            cascades_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]

            # filter cascades by their publish date/time
            # if 'weibo' or 'xovee' in opts.RAWDATA_PATH:
            #     print(opts.RAWDATA_PATH)
            #     exit(0)
            #     # timezone invariant
            #     hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
            #     # 18 for t_o of 0.5 hour and 19 for t_o of 1 hour
            #     if hour < opts.start_hour or hour >= opts.end_hour:
            #         discard_midnight += 1
            #         continue
            # elif 'twitter' in opts.RAWDATA_PATH:
            #     continue
            # elif 'aps' in opts.RAWDATA_PATH:
            #     continue
            # elif 'dblp' in opts.RAWDATA_PATH:
            #     print('tttttt')
            #     exit(0)                         
            msg_time = int(parts[2])
            hour = time.strftime("%H", time.localtime(msg_time))
            hour = int(hour)
            if hour <= opts.start_hour or hour >= opts.end_hour:
                discard_midnight += 1
                continue

            paths = parts[4].strip().split(' ')

            observation_path = list()
            # number of observed popularity
            p_o = 0
            for p in paths:
                # observed adoption/participant
                nodes = p.split(':')[0].split('/')
                time_now = int(p.strip().split(':')[1])
                if time_now < ob_time:
                    p_o += 1
                # save observed adoption/participant into 'observation_path'
                observation_path.append((nodes, time_now))

            # filter cascades which observed popularity less than 10
            if p_o < opts.least_num or p_o > opts.up_num:
                discard_outer += 1
                continue

            # sort list by their publish time/date
            # observation_path.sort(key=lambda tup: tup[1])

            # for each cascade, save its publish time into a dict
            if 'aps' in opts.RAWDATA_PATH:
                cascades_time_dict[cascade_id] = int(0)
            else:
                cascades_time_dict[cascade_id] = int(parts[2])

            o_path = list()

            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                o_path.append('/'.join(nodes) + ':' + str(t))

            # write data into the targeted file, if they are not excluded
            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                   + parts[3] + '\t' + ' '.join(o_path) + '\n'
            filtered_data.append(line)
            cascades_valid_total += 1
    print("total_readin:", cascades_total)
    print("discard_midnight:", discard_midnight)
    print("discard_outer:", discard_outer)
    print('total:', cascades_valid_total)
    # open three files to save train, val, and test set, respectively
    with open(file_train, 'w') as data_train, \
            open(file_val, 'w') as data_val, \
            open(file_test, 'w') as data_test:

        def shuffle_cascades(shuffle=False):
            # shuffle all cascades
            shuffle_time = list(cascades_time_dict.keys())
            # random.seed(seed)
            # random.shuffle(shuffle_time)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(shuffle_time, [None] * len(shuffle_time), test_size=1 - 0.7, random_state=opts.random_seed, shuffle=shuffle)

            val_per = 0.15 / (0.15 + 0.15)

            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_per, random_state=opts.random_seed, shuffle=shuffle)
            for k in X_train:
                cascades_type[k] = 0
            for k in X_val:
                cascades_type[k] = 1
            for k in X_test:
                cascades_type[k] = 2

        shuffle_cascades(shuffle=True)

        # number of valid cascades
        print("Number of valid cascades: {}/{}".format(cascades_valid_total, cascades_total))

        # 3 lists to save the filtered sets
        filtered_data_train = list()
        filtered_data_val = list()
        filtered_data_test = list()
        for line in filtered_data:
            cascade_id = line.split('\t')[0]
            if cascades_type[cascade_id] == 0:
                filtered_data_train.append(line)
            elif cascades_type[cascade_id] == 1:
                filtered_data_val.append(line)
            elif cascades_type[cascade_id] == 2:
                filtered_data_test.append(line)
            else:
                print('What happened?')

        print("Number of valid train cascades: {}".format(len(filtered_data_train)))
        print("Number of valid   val cascades: {}".format(len(filtered_data_val)))
        print("Number of valid  test cascades: {}".format(len(filtered_data_test)))

        # shuffle the train set again
        # random.seed(opts.random_seed)
        # random.shuffle(filtered_data_train)

        def file_write(file_name, cascade_id, observation_path, label):
            # write file, note that compared to the original 'dataset.txt', only cascade_id and each of the
            # observed adoptions are saved, plus label information at last
            file_name.write(cascade_id + '\t' + ''.join(observation_path) + '\t' + label + '\n')

        def graph_write(all_edges, graph_save_path):
            with open(graph_save_path, 'w') as file_name:
                for item in all_edges:
                    file_name.write(item + '\n')

        edges = set()
        # write cascades into files
        for line in filtered_data_train + filtered_data_val + filtered_data_test:
            # split the cascades into 5 parts
            parts = line.split('\t')
            cascade_id = parts[0]
            observation_path = list()
            label = int()

            paths = parts[4].split(' ')

            for p in paths:
                nodes = p.split(':')[0].split('/')
                time_now = int(p.split(':')[1])
                if time_now < ob_time:
                    if time_now == 0:
                        observation_path.append(nodes[-1])
                    else:
                        observation_path.append(' ' + nodes[-1] + ' ' + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ' ' + nodes[i])
                # add label information depends on prediction_time, e.g., 24 hours for weibo dataset
                if time_now < pred_time:
                    label += 1

            # calculate the incremental popularity
            label = str(label - len(observation_path))

            # write files by cascade type
            # 0 to train, 1 to val, 2 to test
            if cascade_id in cascades_type and cascades_type[cascade_id] == 0:
                file_write(data_train, cascade_id, observation_path, label)
            elif cascade_id in cascades_type and cascades_type[cascade_id] == 1:
                file_write(data_val, cascade_id, observation_path, label)
            elif cascade_id in cascades_type and cascades_type[cascade_id] == 2:
                file_write(data_test, cascade_id, observation_path, label)
        # graph include data from train test val
        # graph will be regoconized in the next step
        graph_write(edges, graph_save_path)


def generate_global_graph(file_name):
    # generate_global_graph的时候， 没有筛选-1 没有筛选观测时间，凡是转发，则有图连接，带权
    g = nx.DiGraph()

    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            paths = parts[4].strip().split(' ')
            for path in paths:
                time_now = path.split(':')[-1]
                # discard node after ob
                # NOTE: 可能需要add node 不 add edge
                if int(time_now) > opts.observation_time:
                    continue
                nodes = path.split(':')[0].split('/')
                if len(nodes) < 2:
                    g.add_node(nodes[-1])
                else:
                    g.add_edge(nodes[-1], nodes[-2])

    print("Number of nodes in global graph:", g.number_of_nodes())
    print("Number of edges in global graph:", g.number_of_edges())


def main(argv=None):
    time_start = time.time()
    print('Start to run the TopoLSTM preprocess code!\n')
    print('Dataset path: {}\n'.format(opts.RAWDATA_PATH))

    generate_cascades(
        opts.observation_time,
        opts.prediction_time,
        opts.RAWDATA_PATH,
        opts.DATA_PATH + 'train.txt',
        opts.DATA_PATH + 'val.txt',
        opts.DATA_PATH + 'test.txt',
        opts.DATA_PATH + 'graph.txt',
    )
    summary = False
    if summary == True:
        #summary the graph
        generate_global_graph(opts.RAWDATA_PATH)

    print('Processing time: {:.2f}s'.format(time.time() - time_start))


if __name__ == '__main__':
    main()
