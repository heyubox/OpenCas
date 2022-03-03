from optparse import OptionParser, OptionGroup
import os
import sys
'''
    NOTE: To chage dataset reconifg: 
        1.opts.dataset   a dir where hold dataset to preprocess and train
        2.opts.rawdataset   a file under dir--rawdata follows certain format and cant be compatible between models
'''
op = OptionParser()

# ---------------------config for data processing------------
ge_op = OptionGroup(op, "General Options")

# path
ge_op.add_option("--rawdata_root", dest="rawdata_root", type="string", default="./rawdata/", help="raw dataset root")
ge_op.add_option("--rawdataset", dest="rawdataset", type="string", default="dataset_citation.txt", help="raw data set")
ge_op.add_option("--data_root", dest="data_root", type="string", default="./data/casflow/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="aps/", help="data set.")
ge_op.add_option('--gg_path', dest="gg_path", type="string", default='global_graph.pkl', help='Global graph path.')

# observation and prediction time settings:
# for twitter dataset, we
# for weibo   dataset, we use 1800 (0.5 hour) and 3600 (1 hour) as observation time
#                      we use 3600*24 (86400, 1 day) as prediction time
# for aps     dataset, we
# flags.DEFINE_integer('observation_time', 3600, 'Observation time.')
# flags.DEFINE_integer('prediction_time', 3600*24, 'Prediction time.')
ge_op.add_option('--prediction_time', dest="prediction_time", type="int", default=24 * 3600, help='Number of observation time.')
ge_op.add_option('--observation_time', dest="observation_time", type="int", default=3600, help='Number of observation time.')

# embeddings config
ge_op.add_option('--cg_emb_dim', dest="cg_emb_dim", type="int", default=40, help='Cascade graph embedding dimension.')
ge_op.add_option('--gg_emb_dim', dest="gg_emb_dim", type="int", default=40, help='Global graph embedding dimension.')
ge_op.add_option('--max_seq', dest="max_seq", type="int", default=100, help='Max length of cascade sequence.')
ge_op.add_option('--num_s', dest="num_s", type="int", default=2, help='Number of s for spectral graph wavelets.')
ge_op.add_option("--least_num", dest="least_num", type="int", default=5, help="least num in cascade")
ge_op.add_option("--up_num", dest="up_num", type="int", default=100, help="up num in cascade")
ge_op.add_option("--start_hour", dest="start_hour", type="int", default=7, help="cascade start hour")
ge_op.add_option("--end_hour", dest="end_hour", type="int", default=19, help="cascade end hour")
ge_op.add_option('--random_seed', dest="random_seed", type="int", default=42, help='random_seed.')

# ---------------------config for model training------------

ge_op.add_option("--save_dir", dest="save_dir", type="string", default="../model_save/casflow/", help="model save dir")

op.add_option_group(ge_op)

# ---------------------config for TensorFlow------------

tfop = OptionGroup(op, "TensorFlow Options")

tfop.add_option("--emb_learning_rate", dest="emb_learning_rate", default=10e-3, help="embedding learning_rate.")
tfop.add_option("--learning_rate", dest="learning_rate", default=5e-4, help="learning_rate.")
tfop.add_option("--b_size", dest="b_size", default=64, help="Batch size.")
tfop.add_option("--emb_dim", dest="emb_dim", default=40 + 40, help='Embedding dimension (cascade emb_dim + global emb_dim')
tfop.add_option("--z_dim", dest="z_dim", default=64, help='Dimension of latent variable z.')
tfop.add_option("--rnn_units", dest="rnn_units", default=128, help='Number of RNN units.')
tfop.add_option("--n_flows", dest="n_flows", default=8, help='Number of NF transformations.')
tfop.add_option("--verbose", dest="verbose", default=1, help='Verbose.')
tfop.add_option("--patience", dest="patience", default=10, help='Early stopping patience.')
tfop.add_option("--epochs", dest="epochs", default=15, help='epochs.')

# FLAGS = flags.FLAGS
# flags.DEFINE_float  ('lr', 5e-4, 'Learning rate.')
# flags.DEFINE_integer('b_size', 64, 'Batch size.')
# flags.DEFINE_integer('emb_dim', 40+40, 'Embedding dimension (cascade emb_dim + global emb_dim')
# flags.DEFINE_integer('z_dim', 64, 'Dimension of latent variable z.')

# flags.DEFINE_integer('rnn_units', 128, 'Number of RNN units.')

# flags.DEFINE_integer('n_flows', 8, 'Number of NF transformations.')

# flags.DEFINE_integer('verbose', 2, 'Verbose.')
# flags.DEFINE_integer('patience', 10, 'Early stopping patience.')

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

# opts.up_num = 100
# opts.least_num = 5
if 'citation' in opts.rawdataset:
    opts.observation_time = 360*5
    opts.interval = 180
    opts.least_num = 10
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
'''
    NOTE:default reconfig
'''
opts.RAWDATA_PATH = os.path.join(opts.rawdata_root, opts.rawdataset)
opts.DATA_PATH = os.path.join(opts.data_root, opts.dataset)


def create_dir(dir_str):
    if os.path.exists(dir_str):
        return
    else:
        print('create dir'+dir_str)
        os.makedirs(dir_str)


create_dir(opts.RAWDATA_PATH)
create_dir(opts.data_root)
create_dir(opts.DATA_PATH)
create_dir(opts.save_dir)
