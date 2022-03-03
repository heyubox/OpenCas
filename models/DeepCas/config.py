from optparse import OptionParser, OptionGroup
import os
import sys
'''
    NOTE: To chage dataset reconifg: 
        1.opts.dataset   a dir where hold dataset to preprocess and train
        2.opts.rawdataset   a file under dir--rawdata follows certain format and cant be compatible between models
    TODO:
        make start end hour configurable
'''
op = OptionParser()
# ---------------------config for data processing------------
# NOTE: follow hawkes config
ge_op = OptionGroup(op, "General Options")
ge_op.add_option("--rawdata_root", dest="rawdata_root", type="string", default="./rawdata/", help="raw dataset root")
ge_op.add_option("--rawdataset", dest="rawdataset", type="string", default="dataset_weibo.txt", help="raw data set")
ge_op.add_option("--data_root", dest="data_root", type="string", default="./data/deepcas/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="weibo/", help="data set.")
# data saved for data processing like random walks
ge_op.add_option("--cascade_train_new", dest="cascade_train_new", type="string", default="cascade_train_new.txt", help="formated train data result")
ge_op.add_option("--cascade_val_new", dest="cascade_val_new", type="string", default="cascade_val_new.txt", help="formated val data result")
ge_op.add_option("--cascade_test_new", dest="cascade_test_new", type="string", default="cascade_test_new.txt", help="formated test data result")

ge_op.add_option("--global_graph", dest="global_graph", type="string", default="global_graph.txt", help="formated global graph result")

ge_op.add_option("--start_hour", dest="start_hour", type="int", default=7, help="cascade start hour")
ge_op.add_option("--end_hour", dest="end_hour", type="int", default=19, help="cascade end hour")

# ---------------------config for model training------------
# is_weibo True: weibo dataset, is_weibo False: Citation dataset
ge_op.add_option("--is_weibo", dest="is_weibo", default=True, help="is_weibo True: weibo dataset, is_weibo False: Citation dataset")
ge_op.add_option("--PRETRAIN", dest="PRETRAIN", default=False, help="if load pretrain model")

# parse commandline arguments
ge_op.add_option("--walks_per_graph", dest="walks_per_graph", type="int", default=200, help="number of walks per graph.")
ge_op.add_option("--walk_length", dest="walk_length", type="int", default=10, help="length of each walk.")
ge_op.add_option("--trans_type", dest="trans_type_str", type="string", default="edge", help="Type of function for transition probability: edge, deg, and DEG.")
# node2vec params.
ge_op.add_option("--p", dest="p", type="float", default=1.0, help="Return hyperparameter in node2vec.")
ge_op.add_option("--q", dest="q", type="float", default=1.0, help="Inout hyperparameter in node2vec.")
# word2vec params.
ge_op.add_option('--dimensions', dest="dimensions", type="int", default=50, help='Number of dimensions of embedding.')
ge_op.add_option('--window_size', dest="window_size", type="int", default=10, help='Context size for optimization. Default is 10.')
ge_op.add_option('--iter', dest="iter", default=10, type="int", help='Number of epochs in gensim.word2vec(SGD) Old version')
ge_op.add_option('--emb_epoch', dest="emb_epoch", default=10, type="int", help='Number of epochs in gensim.word2vec(SGD)')
ge_op.add_option('--workers', dest="workers", type="int", default=8, help='Number of parallel workers.')

# set observation_num = -1 if observation_time is used as the stop condition
ge_op.add_option('--observation_num', dest="observation_num", type="int", default=5, help='Number of observation number.')
# op.add_option('--observation_time', dest="observation_time", type=int, default=365*5, help='Number of observation time.')
ge_op.add_option('--observation_time', dest="observation_time", type="int", default=3600*3, help='Number of observation time.')
# tfop.add_option('--interval', dest="interval", type="int", default=180, help='interval')
# op.add_option('--sample_num', dest="sample_num", type=int, default=3000, help='Number of sample number of cascades.')
# op.add_option('--prediction_time', dest="prediction_time", type=int, default=365 * 20, help='Number of observation time.')
ge_op.add_option('--prediction_time', dest="prediction_time", type="int", default=24 * 3600, help='Number of observation time.')

ge_op.add_option('--time_or_number', dest="time_or_number", type="string", default='Time', help='Observation by Time or Num.')
ge_op.add_option("--least_num", dest="least_num", type="int", default=5, help="least num in cascade")
ge_op.add_option("--up_num", dest="up_num", type="int", default=100, help="up num in cascade")
ge_op.add_option("--save_dir", dest="save_dir", type="string", default="../model_save/deepcas/", help="model save dir")
ge_op.add_option('--random_seed', dest="random_seed", type="int", default=42, help='random_seed.')

op.add_option_group(ge_op)

# ---------------------config for TensorFlow------------

tfop = OptionGroup(op, "TensorFlow Options")

tfop.add_option("--learning_rate", dest="learning_rate", default=0.01, help="learning_rate")
tfop.add_option("--emb_learning_rate", dest="emb_learning_rate", default=5e-05, help="emb_learning_rate")
tfop.add_option("--sequence_batch_size", dest="sequence_batch_size", default=20, help="sequence batch size.")
tfop.add_option("--batch_size", dest="batch_size", default=32, help="batch size.")
tfop.add_option("--n_hidden_gru", dest="n_hidden_gru", default=32, help="hidden gru size.")
tfop.add_option("--l1", dest="l1", default=5e-5, help="l1.")
tfop.add_option("--l2", dest="l2", default=1e-8, help="l2.")
tfop.add_option("--l1l2", dest="l1l2", default=1.0, help="l1l2.")
tfop.add_option("--activation", dest="activation", default="relu", help="activation function.")
tfop.add_option("--n_sequences", dest="n_sequences", default=200, help="num of sequences.")
tfop.add_option("--training_iters", dest="training_iters", default=500 * 3200 + 1, help="max training iters.")
tfop.add_option("--display_step", dest="display_step", default=100, help="display step.")
# 注意这里加了一个degree的信息，所以维度+1
tfop.add_option("--embedding_size", dest="embedding_size", default=50, help="embedding size.")
tfop.add_option("--n_input", dest="n_input", default=50, help="n_input size.")
tfop.add_option("--n_steps", dest="n_steps", default=10, help="num of step.")
tfop.add_option("--n_hidden_dense1", dest="n_hidden_dense1", default=64, help="dense1 size.")
tfop.add_option("--n_hidden_dense2", dest="n_hidden_dense2", default=32, help="dense2 size.")
tfop.add_option("--version", dest="version", default="v4", help="data version.")
tfop.add_option("--max_grad_norm", dest="max_grad_norm", default=100, help="gradient clip.")
tfop.add_option("--stddev", dest="stddev", default=0.01, help="initialization stddev.")
tfop.add_option("--dropout_prob", dest="dropout_prob", default=1., help="dropout probability.")
tfop.add_option("--classification", dest="classification", default=False, help="classification or regression.")
tfop.add_option("--n_class", dest="n_class", default=2, help="number of class if do classification.")
tfop.add_option("--one_dense_layer", dest="one_dense_layer", default=False, help="number of dense layer out output.")

op.add_option_group(tfop)

# ---------------------config for other options in opts------------

(opts, args) = op.parse_args()
# print(opts)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)
'''
    NOTE: To chage dataset reconifg: 
        1.opts.dataset   a dir where hold dataset to preprocess and train
        2.opts.rawdataset   a file under dir--rawdata follows certain format and cant be compatible between models
'''
'''
    NOTE:reconfig dataset:
    please reconfig both
'''
# ------------ reconfig opts.rawdataset(file) ----------
# opts.rawdataset = "dataset_weibo.txt"  #default "dataset_weibo.txt"
# opts.rawdataset = "dataset_citation.txt"  #default "dataset_weibo.txt"

# ------------ reconfig opts.dataset(dir) -----------

# opts.dataset = "weibo/"  # default "weibo/"
# opts.dataset = "citation/"  # default "weibo-hawkes/"
opts.save_dir = os.path.join(opts.save_dir, opts.dataset)

'''
    NOTE:reconfig preprocess and train
'''
# opts.learning_rate = 0.001
# opts.emb_learning_rate = 5e-05
# opts.least_num = 5  # default 5
opts.is_weibo = True  # default True
opts.global_file_type = "New"  # New nx.read_edgelist Old: the orignial implementation
opts.prediction_time = 3600 * 24  # default 3600 * 24
if 'citation' in opts.rawdataset:
    opts.observation_time = 360*5
    opts.interval = 180
    opts.least_num = 10  # default 5
    pass
elif 'weibo' in opts.rawdataset:
    # opts.observation_time = 3600
    # opts.interval = 180
    pass
elif 'dblp' in opts.rawdataset:
    # opts.observation_time = 7
    # opts.n_time_interval = opts.observation_time
    # opts.interval = 1
    pass

opts.tail = '_new'  # default '_new'
# opts.tail = ''
'''
    NOTE:default reconfig
'''

opts.RAWDATA_PATH = os.path.join(opts.rawdata_root, opts.rawdataset)

if opts.trans_type_str == "edge":
    opts.trans_type = 0
elif opts.trans_type_str == "deg":
    opts.trans_type = 1
else:
    assert opts.trans_type_str == "DEG", "%s: unseen transition type." % opts.trans_type_str
    opts.trans_type = 2

opts.data_root = os.path.expanduser(opts.data_root)

opts.DATA_PATH = os.path.join(opts.data_root, opts.dataset)
data_path = os.path.join(opts.data_root, opts.dataset)
# opts.global_graph_file = os.path.join(data_path, "global_cas_graph170w_deepcas.txt") # only have records before observation time.
# global_graph_file = os.path.join(data_path, "global_graph_aps_citation.txt") # only have records before observation time.
opts.cascade_train = os.path.join(data_path, opts.cascade_train_new)
opts.cascade_val = os.path.join(data_path, opts.cascade_val_new)
opts.cascade_test = os.path.join(data_path, opts.cascade_test_new)
opts.global_graph_file = os.path.join(data_path, opts.global_graph)


def create_dir(dir_str):
    if os.path.exists(dir_str):
        return
    else:
        print('create dir'+dir_str)
        os.makedirs(dir_str)


create_dir(opts.RAWDATA_PATH)
create_dir(opts.data_root)
create_dir(opts.DATA_PATH)
create_dir(data_path)
create_dir(opts.save_dir)
