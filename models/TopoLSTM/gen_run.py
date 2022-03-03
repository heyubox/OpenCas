from utils import preprocessing
from config import opts
import pickle
import os


if __name__ == "__main__":
    # read argument

    train_nodes, train_labels = preprocessing.process_dataset(opts.DATA_PATH, 'train', maxlen=opts.up_num)
    val_nodes, val_labels = preprocessing.process_dataset(opts.DATA_PATH, 'val', maxlen=opts.up_num)
    test_nodes, test_labels = preprocessing.process_dataset(opts.DATA_PATH, 'test', maxlen=opts.up_num)

    # test_nodes = preprocessing.process_test(opts.DATA_PATH)
    seen_nodes = train_nodes | val_nodes | test_nodes
    print('%d seen nodes.' % len(seen_nodes))
    # write seen nodes into file
    filename = os.path.join(opts.DATA_PATH, 'seen_nodes.txt')
    with open(filename, 'w') as f:
        for v in seen_nodes:
            f.write('%s\n' % v)
    # read graph and node index
    # graph in the train dataset
    G, node_to_index = preprocessing.load_graph(opts.DATA_PATH)

    # transform data into input we need
    train_examples, train_labels = preprocessing.load_examples(opts.DATA_PATH, dataset='train', G=G, node_index=node_to_index, maxlen=opts.up_num)

    val_examples, val_labels = preprocessing.load_examples(opts.DATA_PATH, dataset='val', G=G, node_index=node_to_index, maxlen=opts.up_num)

    test_examples, test_labels = preprocessing.load_examples(opts.DATA_PATH, dataset='test', G=G, node_index=node_to_index, maxlen=opts.up_num)

    print("total train cascade : {}".format(len(train_labels)))
    print("total   val cascade : {}".format(len(val_labels)))
    print("total  test cascade : {}".format(len(test_labels)))

    pickle.dump(train_examples, open(opts.DATA_PATH + "train.pkl", 'wb'))
    pickle.dump(train_labels, open(opts.DATA_PATH + "train_labels.pkl", 'wb'))

    pickle.dump(val_examples, open(opts.DATA_PATH + "val.pkl", 'wb'))
    pickle.dump(val_labels, open(opts.DATA_PATH + "val_labels.pkl", 'wb'))

    pickle.dump(test_examples, open(opts.DATA_PATH + "test.pkl", 'wb'))
    pickle.dump(test_labels, open(opts.DATA_PATH + "test_labels.pkl", 'wb'))

    pickle.dump(node_to_index, open(opts.DATA_PATH + "node_to_index.pkl", 'wb'))
