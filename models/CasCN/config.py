import math
from optparse import OptionParser, OptionGroup
import os
import sys
import pickle
'''
    NOTE: To chage dataset reconifg: 
        1.opts.dataset   a dir where hold dataset to preprocess and train
        2.opts.rawdataset   a file under dir--rawdata follows certain format and cant be compatible between models
'''

op = OptionParser()

# ---------------------config for data processing------------
# NOTE: follow hawkes config
ge_op = OptionGroup(op, "General Options")

ge_op.add_option("--rawdata_root", dest="rawdata_root", type="string", default="./rawdata/", help="raw dataset root")
ge_op.add_option("--rawdataset", dest="rawdataset", type="string", default="dataset_citation.txt", help="raw data set")
ge_op.add_option("--data_root", dest="data_root", type="string", default="./data/cascn/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="aps/", help="data set.")
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
ge_op.add_option("--up_num", dest="up_num", type="int", default=100, help="up num in cascade")
# ge_op.add_option("--global_graph",dest="global_graph",type="string",default="global_graph.txt",help="formated global graph result")

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
ge_op.add_option('--iter', dest="iter", default=5, type="int", help='Number of epochs in SGD')
ge_op.add_option('--workers', dest="workers", type="int", default=8, help='Number of parallel workers.')

# set observation_num = -1 if observation_time is used as the stop condition
ge_op.add_option('--observation_num', dest="observation_num", type="int", default=5, help='Number of observation number.')
ge_op.add_option('--observation_time', dest="observation_time", type="int", default=3600*3, help='Number of observation time.')

# op.add_option('--sample_num', dest="sample_num", type=int, default=3000, help='Number of sample number of cascades.')
ge_op.add_option('--prediction_time', dest="prediction_time", type="int", default=24 * 3600, help='Number of observation time.')

ge_op.add_option('--time_or_number', dest="time_or_number", type="string", default='Time', help='Observation by Time or Num.')
ge_op.add_option("--save_dir", dest="save_dir", type="string", default="../model_save/cascn/", help="model save dir")
ge_op.add_option('--random_seed', dest="random_seed", type="int", default=42, help='random_seed.')

op.add_option_group(ge_op)

# ---------------------config for TensorFlow------------

tfop = OptionGroup(op, "TensorFlow Options")

tfop.add_option('--n_time_interval', dest="n_time_interval", type="int", default=6, help='n_time_interval')
tfop.add_option('--interval', dest="interval", type="int", default=180, help='interval')
tfop.add_option("--n_sequences", dest="n_sequences", default=None, help="num of sequences.")
# NOTE:the same as sequences, the max len of cascade
tfop.add_option("--n_steps", dest="n_steps", default=None, help="num of step.")
tfop.add_option("--num_rnn_layers", dest="num_rnn_layers", default=2, help="number of rnn layers .")
tfop.add_option("--cl_decay_steps", dest="cl_decay_steps", default=1000, help="cl_decay_steps .")
tfop.add_option("--num_kernel", dest="num_kernel", default=2, help="chebyshev .")
tfop.add_option("--learning_rate", dest="learning_rate", default=0.005, help="learning_rate.")
tfop.add_option("--batch_size", dest="batch_size", default=16, help="batch size.")
tfop.add_option("--num_hidden", dest="num_hidden", default=32, help="hidden rnn size.")
tfop.add_option("--use_curriculum_learning", dest="use_curriculum_learning", default=None, help="use_curriculum_learning.")
tfop.add_option("--l1", dest="l1", default=5e-5, help="l1.")
tfop.add_option("--l2", dest="l2", default=1e-3, help="l2.")
tfop.add_option("--l1l2", dest="l1l2", default=1.0, help="l1l2.")
tfop.add_option("--activation", dest="activation", default="relu", help="activation function.")
tfop.add_option("--training_iters", dest="training_iters", default=200 * 3200 + 1, help="max training iters.")
tfop.add_option("--display_step", dest="display_step", default=300, help="display step.")
tfop.add_option("--n_hidden_dense1", dest="n_hidden_dense1", default=32, help="dense1 size.")
tfop.add_option("--n_hidden_dense2", dest="n_hidden_dense2", default=16, help="dense2 size.")
tfop.add_option("--version", dest="version", default="v1", help="data version.")
tfop.add_option("--max_grad_norm", dest="max_grad_norm", default=5, help="gradient clip.")
tfop.add_option("--stddev", dest="stddev", default=0.01, help="initialization stddev.")
# NOTE: feadin the graph should be obtain from max node
tfop.add_option("--feat_in", dest="feat_in", default=None, help="num of feature in")
tfop.add_option("--feat_out", dest="feat_out", default=50, help="num of feature out")
tfop.add_option("--lmax", dest="lmax", default=2, help="max L")
tfop.add_option("--num_nodes", dest="num_nodes", default=None, help="number of max nodes in cascade")  # only support max node=100

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


data_path = os.path.join(opts.data_root, opts.dataset)
if os.path.exists(os.path.join(data_path, opts.information)):
    n_nodes, n_sequences, _, max_node = pickle.load(open(os.path.join(data_path, opts.information), 'rb'))
else:
    n_sequences = opts.up_num
    n_nodes, _, max_node = None, None, None

# opts.learning_rate = 0.0005
opts.up_num = 100  # NOTE:defalut 200 later try 10e3  有时候达不到1000node 实际达到的是n_sequences
# opts.least_num = 5  # default 5
opts.is_weibo = True  # default True
# New nx.read_edgelist Old: the orignial implementation
opts.global_file_type = "New"
if 'citation' in opts.rawdataset:
    opts.observation_time = 360*5
    opts.interval = 180
    opts.least_num = 10  # default 5
    pass
elif 'weibo' in opts.rawdataset:
    # opts.observation_time = 3600*3
    # opts.interval = 180
    pass
elif 'dblp' in opts.rawdataset:
    # opts.observation_time = 5
    # opts.n_time_interval = opts.observation_time
    # opts.interval = 1
    pass
opts.num_nodes = min(opts.up_num, max_node if max_node else 100000000)
opts.n_steps = 100  # n_sequences
opts.feat_in = 100  # max_node  # NOTE must be the same as max node in preprocess
'''
    NOTE:default reconfig
'''
opts.RAWDATA_PATH = os.path.join(opts.rawdata_root, opts.rawdataset)
'''
    NOTE: config of DeepCas
    # if opts.trans_type_str == "edge":
    #     opts.trans_type = 0
    # elif opts.trans_type_str == "deg":
    #     opts.trans_type = 1
    # else:
    #     assert opts.trans_type_str == "DEG", "%s: unseen transition type." % opts.trans_type_str
    #     opts.trans_type = 2
'''
opts.time_interval = math.ceil((opts.observation_time) * 1.0 / opts.n_time_interval)  # 向上取整

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
'''
    NOTE: config of DeepCas
    # opts.global_graph_file = os.path.join(data_path, "global_cas_graph170w_deepcas.txt") # only have records before observation time.
    # global_graph_file = os.path.join(data_path, "global_graph_aps_citation.txt") # only have records before observation time.
    # opts.global_graph_file = os.path.join(data_path,opts.global_graph)
'''
'''
    NOTE: out of date
    # train_pkl = DATA_PATHA+"data_train.pkl"# NOTE opts
    # val_pkl = DATA_PATHA+"data_val.pkl"# NOTE opts
    # test_pkl = DATA_PATHA+"data_test.pkl"# NOTE opts
    # information = DATA_PATHA+"information.pkl"# NOTE opts


    # DATA_PATHA = opts.rawdata_root#"../data/"# NOTE:replace by opts.rawdata_root

    #dataset_name = opts.dataset#'hawkes' # NOTE:replace by opts.dataset



    # weibo
    # cascades  = DATA_PATHA+"dataset_weibo.txt" # NOTE rawdata_root join rawdataset-> RAWDATA_PATH



    # cascade_train =   DATA_PATHA+"cascade_train.txt"# NOTE opts.rawdata_roo join relative
    # cascade_val = DATA_PATHA+"cascade_val.txt"# NOTE opts.rawdata_roo join relative
    # cascade_test = DATA_PATHA+"cascade_test.txt"# NOTE opts.rawdata_roo join relative

    # shortestpath_train = DATA_PATHA+"shortestpath_train.txt"# NOTE opts.rawdata_roo join relative
    # shortestpath_val = DATA_PATHA+"shortestpath_val.txt"# NOTE opts.rawdata_roo join relative
    # shortestpath_test = DATA_PATHA+"shortestpath_test.txt"# NOTE opts.rawdata_roo join relative


    # is_weibo = True # NOTE opts
    # least_num = 10 # NOTE opts
    # up_num = 200 # NOTE opts

    # start_hour = 7 # NOTE opts
    # end_hour = 19 # NOTE opts

    # observation = 1*60*60 -1 # NOTE opts
    # pre_times = [24 * 3600] # NOTE opts


    # train_pkl = DATA_PATHA+"data_train.pkl"# NOTE opts
    # val_pkl = DATA_PATHA+"data_val.pkl"# NOTE opts
    # test_pkl = DATA_PATHA+"data_test.pkl"# NOTE opts
    # information = DATA_PATHA+"information.pkl"# NOTE opts
'''


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
