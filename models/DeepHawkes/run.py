from config import opts
import time
import pickle
from utils.model import SDPP
from utils.utils_func import shuffle_data, get_batch, analysis_data, ProgressBar
from utils.metrics import mSEL, mape_loss_func2, MSEL, accuracy
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math
import os
import sys

sys.path.append('./')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.set_random_seed(0)


n_nodes, n_sequences, n_steps = pickle.load(open(opts.information, 'rb'))
opts.n_sequences = n_sequences
opts.n_steps = n_steps
print("dataset information: nodes:{}, n_sequence:{}, n_steps:{} ".format(n_nodes, n_sequences, n_steps))

opts.dropout_prob = 0.001
opts.learning_rate = 0.005
opts.emb_learning_rate = 0.0005

print("===================configuration===================")
print("dropout prob : ", opts.dropout_prob)
print("l2", opts.l2)
print("learning rate : ", opts.learning_rate)
print("emb_learning_rate : ", opts.emb_learning_rate)
print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
print("observation threshold : ", opts.observation_time)
print("prediction time : ", opts.prediction_time)
print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
print("model save at : {}".format(opts.save_dir))
print("===================configuration===================")

x_train, y_train, sz_train, time_train, rnn_index_train, vocabulary_size = pickle.load(open(opts.train_pkl, 'rb'))
x_test, y_test, sz_test, time_test, rnn_index_test, _ = pickle.load(open(opts.test_pkl, 'rb'))
x_val, y_val, sz_val, time_val, rnn_index_val, _ = pickle.load(open(opts.val_pkl, 'rb'))


analysis_data(y_train, y_test, y_val)
print(len(x_train), len(x_test), len(x_val))

training_iters = opts.training_iters
batch_size = opts.batch_size
display_step = math.ceil(len(x_train) / batch_size)  # display for each epoch
epoch_batch = math.ceil(len(sz_train) / batch_size)

# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = SDPP(opts, sess, n_nodes)


step = 0
best_val_loss = 1000
best_test_loss = 1000


# Keep training until reach max iterations or max_try
train_loss = []
train_mse = []
max_try = 8
patience = max_try

start = time.time()
# 保存模型
saver = tf.train.Saver(max_to_keep=5)
pbar_train = ProgressBar(n_total=math.ceil(training_iters/batch_size), desc="Training")
while step * batch_size < training_iters:
    batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index = get_batch(x_train,
                                                                                             y_train,
                                                                                             sz_train,
                                                                                             time_train,
                                                                                             rnn_index_train,
                                                                                             step,
                                                                                             opts)
    pred = model.train_batch(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)

    loss_train = model.get_error(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)
    train_loss.append(loss_train)
    pbar_train(step, {'loss': train_loss[-1]})
    if (step + 1) % display_step == 0:
        print('finish one epoch, time{}'.format(time.time() - start))

    if (step + 1) % display_step == 0:

        val_loss = []
        val_pred, val_truth = [], []
        pbar_val = ProgressBar(n_total=int(len(y_val) / batch_size), desc="Validating")
        for val_step in range(int(len(y_val) / batch_size)):
            val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index = get_batch(x_val, y_val, sz_val, time_val, rnn_index_val, val_step, opts)
            val_loss.append(model.get_error(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index))
            pbar_val(val_step, {"loss": val_loss[-1]})
            predictions = model.predict(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index)
            if opts.classification:
                predictions = predictions.argmax(axis=-1)
            val_pred.extend(predictions.squeeze().tolist())
            val_truth.extend(val_y.squeeze().tolist())

        test_loss = []
        test_pred, test_truth = [], []
        pbar_test = ProgressBar(n_total=int(len(y_test) / batch_size), desc="Testing")
        for test_step in range(int(len(y_test) / batch_size)):
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_test,
                                                                                                y_test,
                                                                                                sz_test,
                                                                                                time_test,
                                                                                                rnn_index_test,
                                                                                                test_step,
                                                                                                opts)
            test_loss.append(model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
            pbar_test(test_step, {"loss": test_loss[-1]})
            predictions = model.predict(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
            if opts.classification:
                predictions = predictions.argmax(axis=-1)
            test_pred.extend(predictions.squeeze().tolist())
            test_truth.extend(test_y.squeeze().tolist())

        predict_result = []
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            patience = max_try

        if np.mean(test_loss) < best_test_loss:
            saver.save(sess, opts.save_dir, global_step=step)
            best_test_loss = np.mean(test_loss)

        if not opts.classification:
            median_loss = np.median((np.array(test_pred) - np.array(test_truth))**2)
            median_val_loss = np.median((np.array(val_pred) - np.array(val_truth))**2)
            # val metric
            val_mape = mape_loss_func2(val_pred, val_truth)
            val_acc = accuracy(np.array(val_pred), np.array(val_truth), 0.5)
            val_mSEL = mSEL(MSEL(val_pred, val_truth))
            # test metric
            test_mape = mape_loss_func2(test_pred, test_truth)
            test_acc = accuracy(np.array(test_pred), np.array(test_truth), 0.5)
            test_mSEL = mSEL(MSEL(test_pred, test_truth))
        else:
            # acc in actually
            acc_test = np.sum(np.equal(np.array(test_pred), np.array(test_truth))) / len(test_pred)
            acc_valid = np.sum(np.equal(np.array(val_pred), np.array(val_truth))) / len(val_pred)

        if not opts.classification:
            print("\n#" + str(int(step / display_step)) + ", Training Loss= " + "{:.5f}".format(np.mean(train_loss)) + ", Valid Loss= " + "{:.5f}".format(np.mean(val_loss)) + ", Valid median Loss= " +
                  "{:.5f}".format(median_val_loss) + ", Best Valid Loss= " + "{:.5f}".format(best_val_loss) + ", Test Loss= " + "{:.5f}".format(np.mean(test_loss)) + ", Test median  Loss= " +
                  "{:.5f}".format(median_loss) + ", Best Test Loss= " + "{:.5f}".format(best_test_loss))
            print(' Val Loss mSEL:{:5f}, Test Loss mSEL:{:5f},'.format(val_mSEL, test_mSEL))
            print(' Val acc:{:5f}, Test acc:{:5f},'.format(val_acc, test_acc))
            print(' Val Loss mape:{:5f}, Test Loss mape:{:5f},\n'.format(val_mape, test_mape))
        else:
            print("#" + str(int(step / display_step)) + ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + ", Valid acc= " +
                  "{:.6f}".format(acc_valid) + ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) + ", Test acc= " + "{:.6f}".format(acc_test) + ", Best Valid Loss= " +
                  "{:.6f}".format(best_val_loss) + ", Best Test Loss= " + "{:.6f}".format(best_test_loss))

        train_loss = []
        patience -= 1
        if not patience:
            break
    step += 1
    if step % epoch_batch == 0:
        x_train, y_train, sz_train, time_train, rnn_index_train = shuffle_data(x_train, y_train, sz_train, time_train, rnn_index_train)
        x_test, y_test, sz_test, time_test, rnn_index_test = shuffle_data(x_test, y_test, sz_test, time_test, rnn_index_test)
        x_val, y_val, sz_val, time_val, rnn_index_val, = shuffle_data(x_val, y_val, sz_val, time_val, rnn_index_val)


print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
