'''
run cascn
'''
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
from utils.metrics import mape_loss_func2, accuracy, mSEL, MSEL
from utils.utils_func import shuffle_data, get_batch
import sys

sys.path.append('./')


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.set_random_seed(0)

n_nodes, n_sequences, _, max_node = pickle.load(open(opts.information, 'rb'))

# feed in 应该要和下面的一个保持一致
# n_steps=opts.n_steps#min(opts.up_num,n_sequences) # NOTE why not form pkl USED to build Model
# tf.flags.DEFINE_integer("n_sequences", n_sequences, "num of sequences.")
tf.flags.DEFINE_integer("n_steps", opts.n_steps, "num of step.")
tf.flags.DEFINE_integer("time_interval", opts.time_interval, "the time interval")
tf.flags.DEFINE_integer("n_time_interval", opts.n_time_interval, "the number of  time interval")
tf.flags.DEFINE_integer("num_rnn_layers", opts.num_rnn_layers, "number of rnn layers .")
tf.flags.DEFINE_integer("cl_decay_steps", opts.cl_decay_steps, "cl_decay_steps .")
tf.flags.DEFINE_integer("num_kernel", opts.num_kernel, "chebyshev .")
tf.flags.DEFINE_float("learning_rate", opts.learning_rate, "learning_rate.")
tf.flags.DEFINE_integer("batch_size", opts.batch_size, "batch size.")
tf.flags.DEFINE_integer("num_hidden", opts.num_hidden, "hidden rnn size.")
tf.flags.DEFINE_integer("use_curriculum_learning", opts.use_curriculum_learning, "use_curriculum_learning.")
tf.flags.DEFINE_float("l1", opts.l1, "l1.")
tf.flags.DEFINE_float("l2", opts.l2, "l2.")
tf.flags.DEFINE_float("l1l2", opts.l1l2, "l1l2.")
tf.flags.DEFINE_string("activation", opts.activation, "activation function.")
tf.flags.DEFINE_integer("training_iters", opts.training_iters, "max training iters.")
# tf.flags.DEFINE_integer("display_step",opts.display_step, "display step.")
tf.flags.DEFINE_integer("n_hidden_dense1", opts.n_hidden_dense1, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", opts.n_hidden_dense2, "dense2 size.")
tf.flags.DEFINE_string("version", opts.version, "data version.")
tf.flags.DEFINE_integer("max_grad_norm", opts.max_grad_norm, "gradient clip.")
tf.flags.DEFINE_float("stddev", opts.stddev, "initialization stddev.")
# NOTE: Feed_in 992 equal to max node
tf.flags.DEFINE_float("feat_in", opts.feat_in, "num of feature in")
tf.flags.DEFINE_float("feat_out", opts.feat_out, "num of feature out")
tf.flags.DEFINE_float("lmax", opts.lmax, "max L")
tf.flags.DEFINE_float("num_nodes", opts.num_nodes, "number of max nodes in cascade")
# NOTE: num_nodes realte to L relate to max cascade length， feed_in depends on L

tf.flags.DEFINE_string('rawdataset', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('dataset', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('observation_time', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('interval', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('least_num', 'value', 'The explanation of this parameter is ing')
tf.flags.DEFINE_string('up_num', 'value', 'The explanation of this parameter is ing')

print("===================configuration===================")
print("l2", opts.l2)
print("learning rate : ", opts.learning_rate)
print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
print("observation threshold : ", opts.observation_time)
print("prediction time : ", opts.prediction_time)
print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
print("model save at : {}".format(opts.save_dir))
print("===================configuration===================")


version = opts.version
id_train, x_train, L_train, y_train, sz_train, time_train, vocabulary_size, _ = pickle.load(open(opts.train_pkl, 'rb'))
id_test, x_test, L_test, y_test, sz_test, time_test, _, _ = pickle.load(open(opts.test_pkl, 'rb'))
id_val, x_val, L_val, y_val, sz_val, time_val, _, _ = pickle.load(open(opts.val_pkl, 'rb'))

training_iters = opts.training_iters
batch_size = opts.batch_size
# print(len(sz_train) / batch_size)
display_step = int(max(opts.display_step, len(sz_train) / batch_size))
epoch_batch = math.ceil(len(sz_train) / batch_size)

print("-----------------display step-------------------")
print("display step : " + str(display_step))

# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
is_training = False
model = Model(tf.flags.FLAGS, n_nodes, sess)
saver = tf.train.Saver(max_to_keep=5)
# sess.graph.finalize()
step = 0
best_val_loss = 1000
best_test_loss = 1000
# train_writer = tf.summary.FileWriter("./train", sess.graph)

print("building time:", time.time() - start)
# Keep training until reach max iterations or max_try
train_loss = []
max_try = 8
patience = max_try
print("-----------------display step-------------------")
print("display step : " + str(display_step*batch_size))

print("l2", opts.l2)
print("learning rate:", opts.learning_rate)
print('\nfeat_in:', opts.feat_in, "\nnum_nodes:", opts.num_nodes, '\nn_sequences:', n_sequences)
print("n_nodes:  ", n_nodes)

print('-----------start to train--------------')
tbar = tqdm(total=training_iters)
start_100batch = time.time()
while step * batch_size < training_iters:

    step += 1
    batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index = get_batch(x_train, L_train, y_train, sz_train, time_train, step, opts)
    batch_time_interval = np.array(batch_time_interval)
    # print('get a batch data')
    time_decay = model.train_batch(batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index)
    train_loss.append(model.get_error(batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index))
    if step % 100 == -1:
        print('\nfinish training 100 batch:', time.time() - start_100batch, 'loss:', train_loss[-1])
        start_100batch = time.time()
        # saver.save(sess, opts.save_dir, global_step=step)
    if step % display_step == 0:
        cut_size = 1
        print('\ntraing time for an epoch:', time.time() - start, '\n evaluating.........')
        # Calculate batch loss
        # print(time_decay)
        val_predict = []
        val_truth = []
        val_loss = []
        display_bar = tqdm(total=int(len(y_val) / batch_size * cut_size) + int(len(y_test) / batch_size * cut_size + 1))
        for val_step in range(int(len(y_val) / batch_size * cut_size)):
            val_x, val_L, val_y, val_time, val_rnn_index = get_batch(x_val, L_val, y_val, sz_val, time_val, val_step, opts)
            val_predict.extend(model.predict(val_x, val_L, val_y, val_time, val_rnn_index))
            val_truth.extend(val_y)
            val_loss.append(model.get_error(val_x, val_L, val_y, val_time, val_rnn_index))
            display_bar.update(1)

        test_predict = []
        test_truth = []
        test_loss = []
        for test_step in range(int(len(y_test) / batch_size * cut_size + 1)):
            test_x, test_L, test_y, test_time, test_rnn_index = get_batch(x_test, L_test, y_test, sz_test, time_test, test_step,  opts)
            test_predict.extend(model.predict(test_x, test_L, test_y, test_time, test_rnn_index))
            test_truth.extend(test_y)
            test_loss.append(model.get_error(test_x, test_L, test_y, test_time, test_rnn_index))
            display_bar.update(1)

        if np.mean(val_loss) < best_val_loss:
            saver.save(sess, opts.save_dir, global_step=step)
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try
        # val metric
        val_mape = mape_loss_func2(val_predict, val_truth)
        val_acc = accuracy(val_predict, val_truth, 0.5)
        val_mSEL = mSEL(MSEL(val_predict, val_truth))
        # test metric
        test_mape = mape_loss_func2(test_predict, test_truth)
        test_acc = accuracy(test_predict, test_truth, 0.5)
        test_mSEL = mSEL(MSEL(test_predict, test_truth))
        print("\n#" + str(step / display_step) + ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + ", Test Loss= " +
              "{:.6f}".format(np.mean(test_loss)) + ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) + ", Best Test Loss= " + "{:.6f}".format(best_test_loss))
        print(
            ' Val Loss mSEL:{:5f}, Test Loss mSEL:{:5f},'
            .format(val_mSEL, test_mSEL))
        print(
            ' Val acc:{:5f}, Test acc:{:5f},'
            .format(val_acc, test_acc))
        print(
            ' Val Loss mape:{:5f}, Test Loss mape:{:5f},\n'
            .format(val_mape, test_mape))
        print('\nafter evaluating:', time.time() - start)
        start = time.time()
        train_loss = []
        patience -= 1
        if not patience:
            break
    tbar.update(batch_size)
    if step - 1 % epoch_batch == 0:
        x_train, L_train, y_train, sz_train, time_train = shuffle_data(x_train, L_train, y_train, sz_train, time_train)
        x_test, L_test, y_test, sz_test, time_test = shuffle_data(x_test, L_test, y_test, sz_test, time_test)
        x_val, L_val, y_val, sz_val, time_val = shuffle_data(x_val, L_val, y_val, sz_val, time_val)

print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
