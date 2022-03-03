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
ge_op.add_option("--data_root", dest="data_root", type="string", default="./data/cas2vec/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="weibo/", help="data set.")

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
ge_op.add_option("--up_num", dest="up_num", type="int", default=100, help="up num in cascade")
ge_op.add_option('--random_seed', dest="random_seed", type="int", default=42, help='random_seed.')

# ---------------------config for model training------------
# is_weibo True: weibo dataset, is_weibo False: Citation dataset
ge_op.add_option("--is_weibo", dest="is_weibo", default=True, help="is_weibo True: weibo dataset, is_weibo False: Citation dataset")
ge_op.add_option("--PRETRAIN", dest="PRETRAIN", default=False, help="if load pretrain model")
ge_op.add_option("--rank", dest="rank", default=0, help="the gpu for training")

# set observation_num = -1 if observation_time is used as the stop condition
ge_op.add_option('--observation_num', dest="observation_num", type="int", default=5, help='Number of observation number.')
ge_op.add_option('--observation_time', dest="observation_time", type="int", default=3600, help='Number of observation time.')

ge_op.add_option('--prediction_time', dest="prediction_time", type="int", default=24 * 3600, help='Number of observation time.')

ge_op.add_option('--time_or_number', dest="time_or_number", type="string", default='Time', help='Observation by Time or Num.')
ge_op.add_option("--save_dir", dest="save_dir", type="string", default="../model_save/cas2vec/", help="model save dir")

op.add_option_group(ge_op)
# ---------------------config for TensorFlow------------

tfop = OptionGroup(op, "TensorFlow/Pytorch Options")

tfop.add_option('--n_time_interval', dest="n_time_interval", type="int", default=6, help='n_time_interval')
tfop.add_option('--interval', dest="interval", type="int", default=180, help='sample interval ')
tfop.add_option("--learning_rate", dest="learning_rate", default=0.001, help="learning_rate.")
tfop.add_option("--batch_size", dest="batch_size", type="int", default=64, help="batch size.")
tfop.add_option("--hidden_size", dest="hidden_size", default=64, help="hidden size.")
tfop.add_option("--middle_size", dest="middle_size", default=64, help="middle_size.")
tfop.add_option("--gnn_out_features", dest="gnn_out_features", default=64, help="hidden size.")
tfop.add_option("--num_mlp_layers", dest="num_mlp_layers", default=1, help="num_mlp_layers.")
tfop.add_option("--num_layers", dest="num_layers", default=2, help="num_layers.")
tfop.add_option("--mlp_hidden", dest="mlp_hidden", default=32, help="mlp_hidden.")
tfop.add_option("--input_features", dest="input_features", default=100, help="input_features.")
tfop.add_option("--out_features", dest="out_features", default=1, help="out_features.")
tfop.add_option('--epochs', dest="epochs", default=50, type="int", help='Number of epochs in training')
tfop.add_option('--gnn_mlp_hidden', dest="gnn_mlp_hidden", type="int", default=32, help='gnn_mlp_hidden kernal size')
tfop.add_option('--windows_size', dest="windows_size", type="int", default=5, help='Context size for optimization.')
tfop.add_option('--bidirection', dest="bidirection", type="string",  default=True, help='bidirection settings of gru/lstm.')
tfop.add_option('--attention_type', dest="attention_type",  type="string", default='GAT', help='attention_type ,no attention when set to None.choice:[GAT/None].')
tfop.add_option('--gcn_type', dest="gcn_type",  type="string", default='self', help='gcn_type: self/gcn.')
tfop.add_option('--rnn_type', dest="rnn_type",  type="string", default='gru', help='rnn_type: gru/lstm.')


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
# opts.up_num = 100
# opts.least_num = 5  # default 5
opts.prediction_time = 3600 * 24  # default 3600 * 24
# if 'citation' in opts.rawdataset:
#     opts.observation_time = 360*5
#     opts.interval = 180
# elif 'weibo' in opts.rawdataset:
#     opts.observation_time = 3600*3
#     opts.interval = 360
# elif 'dblp' in opts.rawdataset:
#     opts.observation_time = 7
#     opts.n_time_interval = opts.observation_time
#     opts.interval = 1
# opts.batch_size = 64
# opts.epochs=30

'''
    NOTE:default reconfig
'''
opts.RAWDATA_PATH = os.path.join(opts.rawdata_root, opts.rawdataset)

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

if opts.bidirection == 'True':
    opts.bidirection = True
elif opts.bidirection == 'False':
    opts.bidirection = False


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
