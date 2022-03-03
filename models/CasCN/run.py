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
import time
from config import opts
from utils.metrics import mape_loss_func2, accuracy, mSEL, MSEL
from utils.utils_func import shuffle_data, get_batch, ProgressBar
import sys

sys.path.append('./')


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.set_random_seed(0)

n_nodes, n_sequences, _, max_node = 100, 100, 100, 100  # pickle.load(open(opts.information, 'rb'))

print("===================configuration===================")
print("l2", opts.l2)
print("learning rate : ", opts.learning_rate)  # 0.005 in the paper
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
model = Model(opts, n_nodes, sess)
saver = tf.train.Saver(max_to_keep=5)
step = 0
best_val_loss = 1000
best_test_loss = 1000

print("building time:", time.time() - start)
# Keep training until reach max iterations or max_try
train_loss = []
max_try = 8
patience = max_try

print("l2", opts.l2)
print("learning rate:", opts.learning_rate)
print('\nfeat_in:', opts.feat_in, "\nnum_nodes:", opts.num_nodes, '\nn_sequences:', n_sequences)
print("n_nodes:  ", n_nodes)

print('-----------start to train--------------')
start_100batch = time.time()
pbar_train = ProgressBar(n_total=math.ceil(training_iters/batch_size), desc='Training')
while step * batch_size < training_iters:

    try:
        step += 1
        batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index = get_batch(x_train, L_train, y_train, sz_train, time_train, step, opts)
        batch_time_interval = np.array(batch_time_interval)
        # print('get a batch data')
        time_decay = model.train_batch(batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index)
        train_loss.append(model.get_error(batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index))
        pbar_train(step, {'loss': train_loss[-1]})
    except Exception as e:
        print(e)
    if step % display_step == 0:
        cut_size = 1
        # Calculate batch loss
        val_predict = []
        val_truth = []
        val_loss = []
        pbar_val = ProgressBar(n_total=int(len(y_val) / batch_size * cut_size + 1), desc='Validating')
        pbar_test = ProgressBar(n_total=int(len(y_test) / batch_size * cut_size + 1), desc='Testing')
        for val_step in range(int(len(y_val) / batch_size * cut_size)):
            val_x, val_L, val_y, val_time, val_rnn_index = get_batch(x_val, L_val, y_val, sz_val, time_val, val_step, opts)
            val_predict.extend(model.predict(val_x, val_L, val_y, val_time, val_rnn_index))
            val_truth.extend(val_y)
            val_loss.append(model.get_error(val_x, val_L, val_y, val_time, val_rnn_index))
            pbar_val(val_step, {'loss': val_loss[-1]})

        test_predict = []
        test_truth = []
        test_loss = []
        for test_step in range(int(len(y_test) / batch_size * cut_size + 1)):
            test_x, test_L, test_y, test_time, test_rnn_index = get_batch(x_test, L_test, y_test, sz_test, time_test, test_step,  opts)
            test_predict.extend(model.predict(test_x, test_L, test_y, test_time, test_rnn_index))
            test_truth.extend(test_y)
            test_loss.append(model.get_error(test_x, test_L, test_y, test_time, test_rnn_index))
            pbar_test(test_step, {'loss': test_loss[-1]})

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
    # tbar.update(batch_size)
    if step - 1 % epoch_batch == 0:
        x_train, L_train, y_train, sz_train, time_train = shuffle_data(x_train, L_train, y_train, sz_train, time_train)
        x_test, L_test, y_test, sz_test, time_test = shuffle_data(x_test, L_test, y_test, sz_test, time_test)
        x_val, L_val, y_val, sz_val, time_val = shuffle_data(x_val, L_val, y_val, sz_val, time_val)

print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
