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
# ge_op.add_option("--rawdataset", dest="rawdataset", type="string", default="dataset_weibo.txt", help="raw data set")
ge_op.add_option("--data_root", dest="data_root", type="string", default="./data/topolstm/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="aps/", help="data set.")
ge_op.add_option('--graph', dest="graph", type="string", default='graph.txt', help='Seen graph path.')


ge_op.add_option('--prediction_time', dest="prediction_time", type="int", default=24 * 3600, help='Number of observation time.')
ge_op.add_option('--observation_time', dest="observation_time", type="int", default=3600, help='Number of observation time.')

# embeddings config
ge_op.add_option("--least_num", dest="least_num", type="int", default=5, help="least num in cascade")
ge_op.add_option("--up_num", dest="up_num", type="int", default=100, help="up num in cascade")
ge_op.add_option("--start_hour", dest="start_hour", type="int", default=7, help="cascade start hour")
ge_op.add_option("--end_hour", dest="end_hour", type="int", default=19, help="cascade end hour")
ge_op.add_option('--random_seed', dest="random_seed", type="int", default=42, help='random_seed.')

# ---------------------config for model training------------

ge_op.add_option("--save_dir", dest="save_dir", type="string", default="../model_save/topolstm/", help="model save dir")

op.add_option_group(ge_op)

# ---------------------config for TensorFlow------------

tfop = OptionGroup(op, "TensorFlow Options")

tfop.add_option("--learning_rate", dest="learning_rate", default=1e-3, help="learning_rate.")
tfop.add_option("--batch_size", dest="batch_size", default=256, help="Batch size.")
tfop.add_option('--hidden_size', dest="hidden_size", default=64, type=int, help='embedding size')
tfop.add_option("--epochs", dest="epochs", default=40, help='epochs.')

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
# opts.least_num = 10  # default 5
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
    # opts.observation_time = 6
    # opts.n_time_interval = opts.observation_time
    # opts.interval = 1
    pass


'''
    NOTE:default reconfig
'''
opts.RAWDATA_PATH = os.path.join(opts.rawdata_root, opts.rawdataset)
opts.DATA_PATH = os.path.join(opts.data_root, opts.dataset)
print('===========configuration loading success==========')


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
