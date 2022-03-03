from config import opts
import time
import pickle
from utils.model import SDPP
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math
import os
import sys

sys.path.append('./')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('os.environ:', os.environ['CUDA_VISIBLE_DEVICES'])
# tf.reset_default_graph()
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0

tf.set_random_seed(0)

NUM_THREADS = 20
# DATA_PATH = "data"

print('ok')
opts.batch_size = 1
n_nodes, n_sequences, n_steps = pickle.load(open(opts.information, 'rb'))
print("dataset information: nodes:{}, n_sequence:{}, n_steps:{} ".format(n_nodes, n_sequences, n_steps))
tf.flags.DEFINE_integer("n_sequences", n_sequences, "num of sequences.")
tf.flags.DEFINE_integer("n_steps", n_steps, "num of step.")
tf.flags.DEFINE_integer("time_interval", opts.time_interval, "the time interval")
tf.flags.DEFINE_integer("n_time_interval", opts.n_time_interval, "the number of  time interval")

tf.flags.DEFINE_float("learning_rate", opts.learning_rate, "learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", opts.sequence_batch_size, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size", opts.batch_size, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", opts.n_hidden_gru, "hidden gru size.")
tf.flags.DEFINE_float("l1", opts.l1, "l1.")
tf.flags.DEFINE_float("l2", opts.l2, "l2.")
tf.flags.DEFINE_float("l1l2", opts.l1l2, "l1l2.")
tf.flags.DEFINE_string("activation", opts.activation, "activation function.")
tf.flags.DEFINE_integer("training_iters", opts.training_iters, "max training iters.")
tf.flags.DEFINE_integer("display_step", opts.display_step, "display step.")
tf.flags.DEFINE_integer("embedding_size", opts.embedding_size, "embedding size.")
tf.flags.DEFINE_integer("n_input", opts.n_input, "input size.")
tf.flags.DEFINE_integer("n_hidden_dense1", opts.n_hidden_dense1, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", opts.n_hidden_dense2, "dense2 size.")
tf.flags.DEFINE_string("version", opts.version, "data version.")
tf.flags.DEFINE_integer("max_grad_norm", opts.max_grad_norm, "gradient clip.")
tf.flags.DEFINE_float("stddev", opts.stddev, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", opts.emb_learning_rate, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", opts.dropout, "dropout probability.")
tf.flags.DEFINE_boolean("PRETRAIN", opts.PRETRAIN, "Loading PRETRAIN models or not.")
tf.flags.DEFINE_boolean("fix", opts.fix, "Fix the pretrained embedding or not.")
tf.flags.DEFINE_boolean("classification", opts.classification, "classification or regression.")
tf.flags.DEFINE_integer("n_class", opts.n_class, "number of class if do classification.")
tf.flags.DEFINE_boolean("one_dense_layer", opts.one_dense_layer, "number of dense layer out output.")
tf.flags.DEFINE_string('rawdataset', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('dataset', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('observation_time', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('interval', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('least_num', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('up_num', 'value', 'The explanation of this parameter is ing')

config = tf.flags.FLAGS
config.dropout_prob = 0.001
config.learning_rate = 0.001
config.emb_learning_rate = 0.001

print("===================configuration===================")
print("dropout prob : ", config.dropout_prob)
print("l2", config.l2)
print("learning rate : ", config.learning_rate)
print("emb_learning_rate : ", config.emb_learning_rate)
print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
print("observation threshold : ", opts.observation_time)
print("prediction time : ", opts.prediction_time)
print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
print("model save at : {}".format(opts.save_dir))
print("===================configuration===================")


def get_y_label(batch_y):
    y_label = np.zeros_like(batch_y)
    y_label[(batch_y > 0) & (batch_y <= np.log(10. + 1.))] = 0
    batch_y[(batch_y > 0) & (batch_y <= np.log(10. + 1.))] = 0
    y_label[(batch_y > 0) & (batch_y <= np.log(50. + 1.))] = 1
    batch_y[(batch_y > 0) & (batch_y <= np.log(50. + 1.))] = 0
    y_label[(batch_y > 0) & (batch_y <= np.log(100. + 1.))] = 2
    batch_y[(batch_y > 0) & (batch_y <= np.log(100. + 1.))] = 0
    y_label[(batch_y > 0) & (batch_y <= np.log(500. + 1.))] = 3
    batch_y[(batch_y > 0) & (batch_y <= np.log(500. + 1.))] = 0
    y_label[batch_y > 0] = 4
    return y_label


# (total_number of sequence,n_steps)
def get_batch(x, y, sz, time, rnn_index, n_time_interval, step, batch_size=128):
    batch_y = np.zeros(shape=(batch_size, 1))
    batch_x = []
    batch_x_indict = []
    batch_time_interval_index = []
    batch_rnn_index = []
    start = step * batch_size % len(x)

    # print start
    for i in range(batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        for j in range(sz[id]):
            batch_x.append(x[id][j])
            # time_interval
            temp_time = np.zeros(shape=(n_time_interval))
            k = int(math.floor(time[id][j] / config.time_interval))
            # in observation_num model, the k can be larger than n_time_interval
            if k >= config.n_time_interval:
                k = config.n_time_interval - 1

            temp_time[k] = 1
            batch_time_interval_index.append(temp_time)

            # rnn index
            temp_rnn = np.zeros(shape=(config.n_steps))
            if rnn_index[id][j] - 1 >= 0:
                temp_rnn[rnn_index[id][j] - 1] = 1
            batch_rnn_index.append(temp_rnn)

            for k in range(2 * config.n_hidden_gru):
                batch_x_indict.append([i, j, k])

    if config.classification:
        batch_y = get_y_label(batch_y)

    return batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index


version = config.version

x_test, y_test, sz_test, time_test, rnn_index_test, _ = pickle.load(open(opts.test_pkl, 'rb'))
x_val, y_val, sz_val, time_val, rnn_index_val, _ = pickle.load(open(opts.val_pkl, 'rb'))


# do analysis
def analysis_data(Y_train, Y_test, Y_valid):
    print('---------***---------')
    print("Number: {}, Max: {} and Min: {} label Value in Train".format(len(Y_train), max(Y_train), min(Y_train)))
    print('NUmber: {}, Max: {} and Min: {} label Value in Test'.format(len(Y_test), max(Y_test), min(Y_test)))
    print('NUmber: {}, Max: {} and Min: {} label Value in Valid'.format(len(Y_valid), max(Y_valid), min(Y_valid)))
    print('---------***---------')


def shuffle_data(x, y, wz, time, rnn_index):
    seed = np.random.randint(low=0, high=100)
    return shuffle(x, y, wz, time, rnn_index, random_state=seed)


analysis_data([0], y_test, y_val)
print(len([0]), len(x_test), len(x_val))

training_iters = config.training_iters
batch_size = config.batch_size
# display_step = min(config.display_step, len(sz_train) / batch_size)
# display_step = 5
# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

model_save_path = opts.save_dir


def mape_loss_func2(preds, labels):
    preds = np.clip(np.array(preds), 1, 1000)
    labels = np.clip(np.array(labels), 1, 1000)
    return np.fabs((labels-preds)/labels).mean()


def accuracy(preds, labels, bias=0.2):
    preds += 0.01
    labels += 0.01
    diff = np.abs(preds-labels)
    count = labels*bias > diff
    acc = np.sum(count)/len(count)
    return acc


def mSEL(loss):
    loss = np.array(loss)
    loss = loss.flatten()
    return np.median(loss)


def MSEL(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return np.square(preds-labels)


x_test+x_val
preds = []
truth = []
all_hs = []
all_hgs = []
loss = []
all_hs2, all_hgs2 = [], []
embs = None
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model = SDPP(config, sess, n_nodes)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    # 载入模型，不需要提供模型的名字，会通过 checkpoint 文件定位到最新保存的模型
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("\n==============load success====================\n")
        print(model_save_path)
    for iter in tqdm(range(int(len(y_test) / batch_size)), ncols=100):

        test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_test, y_test, sz_test, time_test, rnn_index_test, config.n_time_interval, iter, batch_size=batch_size)
        hs = model.get_hidden_states(
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        # embs = model.get_embedding(
        #     test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        hgs = model.get_hidden_graph_states(
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        test_loss = model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        p = model.predict(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        preds.extend(p[0])
        truth.extend(test_y[0])
        loss.append(test_loss)
        all_hs.append(hs)
        all_hgs.append(hgs)

        test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_val, y_val, sz_val, time_val, rnn_index_val, config.n_time_interval, iter, batch_size=batch_size)
        hs = model.get_hidden_states(
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)

        hgs = model.get_hidden_graph_states(
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        test_loss = model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        p = model.predict(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
        # preds.extend(p[0])
        # truth.extend(test_y[0])
        loss.append(test_loss)
        all_hs2.append(hs)
        all_hgs2.append(hgs)
    embs = model.get_embedding(
        test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)

    print('==========loss: {}==========='.format(np.mean(loss)))
    preds = np.array(preds)
    truth = np.array(truth)
    print('max preds {}, max truth {}'.format(np.max(preds), np.max(truth)))
    print('min preds {}, min truth {}'.format(np.min(preds), np.min(truth)))
    print("mape", mape_loss_func2(preds, truth))
    print("acc", accuracy(preds, truth, 0.5))
    print("MSLE", np.mean(MSEL(preds, truth)))
    data = {
        'Xtest': x_test+x_val,
        'Ytest': y_test+y_val,
        'hidden_states': all_hs+all_hs2,
        "hidden_states_decay": all_hgs+all_hgs2,
        "time": time_test+time_val,
        'emb': embs,
    }

    pickle.dump(data, open(opts.DATA_PATH+'ana.pkl', 'wb'))
