import math
from optparse import OptionParser, OptionGroup
import os
import sys
'''
    NOTE: To chage dataset reconifg: 
        1.opts.dataset   a dir where hold dataset to preprocess and train
        2.opts.rawdataset   a file under dir--rawdata follows certain format and cant be compatible between models
    TODO:
        make start end hour configurable
        default="weibohawkes/" -> default="weibo-hawkes/"
'''
op = OptionParser()

# ---------------------config for data processing------------

ge_op = OptionGroup(op, "General Options")
ge_op.add_option("--rawdata_root", dest="rawdata_root", type="string", default="./rawdata/", help="raw dataset root")
ge_op.add_option("--rawdataset", dest="rawdataset", type="string", default="dataset_weibo.txt", help="raw data set")
ge_op.add_option("--data_root", dest="data_root", type="string", default="./data/deephawkes/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="weibo/", help="data set.")

# data saved for data processing like random walks
ge_op.add_option("--cascade_train", dest="cascade_train", type="string", default="cascade_train.txt", help="formated train data result")
ge_op.add_option("--cascade_val", dest="cascade_val", type="string", default="cascade_val.txt", help="formated val data result")
ge_op.add_option("--cascade_test", dest="cascade_test", type="string", default="cascade_test.txt", help="formated test data result")

ge_op.add_option("--shortestpath_train", dest="shortestpath_train", type="string", default="shortestpath_train.txt", help="formated train data result")
ge_op.add_option("--shortestpath_val", dest="shortestpath_val", type="string", default="shortestpath_val.txt", help="formated val data result")
ge_op.add_option("--shortestpath_test", dest="shortestpath_test", type="string", default="shortestpath_test.txt", help="formated test data result")

ge_op.add_option("--data_train", dest="data_train", type="string", default="data_train.pkl", help="data_train")
ge_op.add_option("--data_val", dest="data_val", type="string", default="data_val.pkl", help="data_val")
ge_op.add_option("--data_test", dest="data_test", type="string", default="data_test.pkl", help="data_test")
ge_op.add_option("--information", dest="information", type="string", default="information.pkl", help="information")
ge_op.add_option("--start_hour", dest="start_hour", type="int", default=7, help="cascade start hour")
ge_op.add_option("--end_hour", dest="end_hour", type="int", default=19, help="cascade end hour")
ge_op.add_option("--least_num", dest="least_num", type="int", default=5, help="least num in cascade")
ge_op.add_option("--up_num", dest="up_num", type="int", default=1000, help="up num in cascade")

ge_op.add_option("--save_dir", dest="save_dir", type="string", default="../model_save/deephawkes/", help="model save dir")

# ---------------------config for model training------------
# is_weibo True: weibo dataset, is_weibo False: Citation dataset
ge_op.add_option("--is_weibo", dest="is_weibo", default=True, help="is_weibo True: weibo dataset, is_weibo False: Citation dataset")
ge_op.add_option("--PRETRAIN", dest="PRETRAIN", default=False, help="if load pretrain model")

# parse commandline arguments
ge_op.add_option("--walks_per_graph", dest="walks_per_graph", type="int", default=200, help="number of walks per graph.")
ge_op.add_option("--walk_length", dest="walk_length", type="int", default=10, help="length of each walk.")
ge_op.add_option("--trans_type", dest="trans_type_str", type="string", default="edge", help="Type of function for transition probability: edge, deg, and DEG.")


# set observation_num = -1 if observation_time is used as the stop condition
ge_op.add_option('--observation_num', dest="observation_num", type="int", default=5, help='Number of observation number.')

ge_op.add_option('--observation_time', dest="observation_time", type="int", default=3600, help='Number of observation time.')

ge_op.add_option('--prediction_time', dest="prediction_time", type="int", default=24 * 3600, help='Number of observation time.')

ge_op.add_option('--time_or_number', dest="time_or_number", type="string", default='Time', help='Observation by Time or Num.')
ge_op.add_option('--random_seed', dest="random_seed", type="int", default=42, help='random_seed.')

op.add_option_group(ge_op)
# ---------------------config for TensorFlow------------

tfop = OptionGroup(op, "TensorFlow Options")

tfop.add_option('--n_time_interval', dest="n_time_interval", type="int", default=6, help='n_time_interval')
tfop.add_option('--interval', dest="interval", type="int", default=180, help='interval')
tfop.add_option("--emb_learning_rate", dest="emb_learning_rate", default=0.003, help="embedding learning_rate.")
tfop.add_option("--learning_rate", dest="learning_rate", default=0.003, help="learning_rate.")
tfop.add_option("--sequence_batch_size", dest="sequence_batch_size", default=20, help="sequence batch size.")
tfop.add_option("--batch_size", dest="batch_size", default=128, help="batch size.")
tfop.add_option("--n_hidden_gru", dest="n_hidden_gru", default=32, help="hidden gru size.")
tfop.add_option("--l1", dest="l1", default=5e-5, help="l1.")
tfop.add_option("--l2", dest="l2", default=0.05, help="l2.")
tfop.add_option("--l1l2", dest="l1l2", default=1.0, help="l1l2.")
tfop.add_option("--activation", dest="activation", default="relu", help="activation function.")
tfop.add_option("--training_iters", dest="training_iters", default=200 * 3200 + 1, help="max training iters.")
tfop.add_option("--display_step", dest="display_step", default=100, help="display step.")
tfop.add_option("--embedding_size", dest="embedding_size", default=50, help="embedding size.")
tfop.add_option("--n_input", dest="n_input", default=50, help="input size.")
tfop.add_option("--n_hidden_dense1", dest="n_hidden_dense1", default=32, help="dense1 size.")
tfop.add_option("--n_hidden_dense2", dest="n_hidden_dense2", default=32, help="dense2 size.")
tfop.add_option("--version", dest="version", default="v4", help="data version.")
tfop.add_option("--max_grad_norm", dest="max_grad_norm", default=100, help="gradient clip.")
tfop.add_option("--stddev", dest="stddev", default=0.01, help="initialization stddev.")
tfop.add_option("--dropout", dest="dropout", default=1., help="dropout probability.")
tfop.add_option("--fix", dest="fix", default=False, help="Fix the pretrained embedding or not.")
tfop.add_option("--classification", dest="classification", default=False, help="classification or regression.")
tfop.add_option("--n_class", dest="n_class", default=5, help="number of class if do classification.")
tfop.add_option("--one_dense_layer", dest="one_dense_layer", default=False, help="number of dense layer out output.")


op.add_option_group(tfop)

# ---------------------config for other options in opts------------

(opts, args) = op.parse_args()
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
# opts.rawdataset = "dataset_dblp.txt"

# ------------ reconfig opts.dataset(dir) -----------

# opts.dataset = "weibo/"  # default "weibo/"
# opts.dataset = "citation/"  # default "weibo-hawkes/"
# opts.dataset = "dblp/"

opts.save_dir = os.path.join(opts.save_dir, opts.dataset)
'''
    NOTE:reconfig preprocess and train
'''
# opts.up_num = 1000
# opts.least_num = 5  # default 5
# New nx.read_edgelist Old: the orignial implementation
opts.global_file_type = "New"
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
# opts.tail = '_new' # default '_new'
# opts.tail = ''
'''
    NOTE:default reconfig
'''
opts.RAWDATA_PATH = os.path.join(opts.rawdata_root, opts.rawdataset)

# some times observation_time+=1
opts.time_interval = math.ceil((opts.observation_time) * 1.0 / opts.n_time_interval)

opts.data_root = os.path.expanduser(opts.data_root)
opts.DATA_PATH = os.path.join(opts.data_root, opts.dataset)
data_path = os.path.join(opts.data_root, opts.dataset)

opts.cascade_train = os.path.join(data_path, opts.cascade_train)
opts.cascade_val = os.path.join(data_path, opts.cascade_val)
opts.cascade_test = os.path.join(data_path, opts.cascade_test)

opts.shortestpath_train = os.path.join(data_path, opts.shortestpath_train)
opts.shortestpath_val = os.path.join(data_path, opts.shortestpath_val)
opts.shortestpath_test = os.path.join(data_path, opts.shortestpath_test)

opts.train_pkl = os.path.join(data_path, opts.data_train)
opts.val_pkl = os.path.join(data_path, opts.data_val)
opts.test_pkl = os.path.join(data_path, opts.data_test)
opts.information = os.path.join(data_path, opts.information)


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
