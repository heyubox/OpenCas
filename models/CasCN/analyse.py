import os
import math
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from utils.model_sparse_graph_signal import Model
import pickle
from tqdm import tqdm
import time
from config import opts
import sys
from utils.metrics import mSEL, mape_loss_func2, MSEL, accuracy
from utils.utils_func import get_batch
sys.path.append('./')


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.set_random_seed(0)

n_nodes, n_sequences, _, max_node = pickle.load(open(opts.information, 'rb'))
opts.batch_size = 1


print("===================configuration===================")
print("l2", opts.l2)
print("learning rate : ", opts.learning_rate)
print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
print("observation threshold : ", opts.observation_time)
print("prediction time : ", opts.prediction_time)
print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
print("model save at : {}".format(opts.save_dir))
print("===================configuration===================")


id_test, x_test, L_test, y_test, sz_test, time_test, _, test_idmap = pickle.load(open(opts.test_pkl, 'rb'))
id_val, x_val, L_val, y_val, sz_val, time_val, _, val_idmap = pickle.load(open(opts.val_pkl, 'rb'))

training_iters = opts.training_iters
batch_size = 1


# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)
model_save_path = opts.save_dir
all_hs = []
loss = []
preds = []
truth = []
all_gnn_o = []
X = []
Y = []
idmap = test_idmap+val_idmap
print('total', len(x_test))
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    model = Model(opts, n_nodes, sess)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("\n==============load success====================\n")
    for iter in tqdm(range(len(x_test)), ncols=100):
        test_x, test_L, test_y, test_time, test_rnn_index = get_batch(x_test, L_test, y_test, sz_test, time_test, iter,  opts)
        hiddens = model.get_rnn_hidden(test_x, test_L, test_y, test_time, test_rnn_index)
        loss.append(model.get_error(test_x, test_L, test_y, test_time, test_rnn_index))
        p = model.predict(test_x, test_L, test_y, test_time, test_rnn_index)
        X.extend(test_x)
        Y.extend(test_y)
        all_gnn_o.append(model.get_gnn_o(test_x, test_L, test_y, test_time, test_rnn_index))
        preds.extend(p[0])
        truth.extend(test_y[0])
        all_hs.append(hiddens.tolist())

    print('==========loss: {}==========='.format(np.mean(loss)))
    preds_ = np.array(preds)
    truth_ = np.array(truth)
    print('max preds {}, max truth {}'.format(np.max(preds_), np.max(truth_)))
    print('min preds {}, min truth {}'.format(np.min(preds_), np.min(truth_)))
    print("mape", mape_loss_func2(preds_, truth_))
    print("acc", accuracy(preds_, truth_, 0.5))

    for iter in tqdm(range(len(x_val)), ncols=100):

        test_x, test_L, test_y, test_time, test_rnn_index = get_batch(x_val, L_val, y_val, sz_val, time_val, iter, opts)
        hiddens = model.get_rnn_hidden(test_x, test_L, test_y, test_time, test_rnn_index)
        loss.append(model.get_error(test_x, test_L, test_y, test_time, test_rnn_index))
        p = model.predict(test_x, test_L, test_y, test_time, test_rnn_index)
        X.extend(test_x)
        Y.extend(test_y)
        all_gnn_o.append(model.get_gnn_o(test_x, test_L, test_y, test_time, test_rnn_index))
        preds.extend(p[0])
        truth.extend(test_y[0])
        all_hs.append(hiddens.tolist())

    print('==========loss: {}==========='.format(np.mean(loss)))
    preds = np.array(preds)
    truth = np.array(truth)
    print('max preds {}, max truth {}'.format(np.max(preds), np.max(truth)))
    print('min preds {}, min truth {}'.format(np.min(preds), np.min(truth)))
    print("mape", mape_loss_func2(preds, truth))
    print("acc", accuracy(preds, truth, 0.5))

    data = {
        # 'Xtest': X[:10000],
        # 'Ytest': Y[:10000],
        # 'hidden_states': all_hs[:10000],
        # 'time':time_test[:10000],
        # 'size':sz_test[:10000],
        'gnn_o': all_gnn_o[:10000],
        'id': idmap[:10000]
    }

    pickle.dump(data, open('ana.pkl', 'wb'))
