import math
from config import opts
import time
import gzip
import pickle
from sklearn.utils import shuffle
from utils.model import DeepCas
import tensorflow as tf
import os
import numpy as np
import sys
from utils.metrics import accuracy, mape_loss_func2, mSEL, MSEL
from utils.utils_func import shuffle_data, get_batch, ProgressBar
sys.path.append('./')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


tf.set_random_seed(0)

print("===================configuration===================")
print("dropout prob:", opts.dropout_prob)
print("l2", opts.l2)
print("learning rate:", opts.learning_rate)
print("emb_learning_rate:", opts.emb_learning_rate)
print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
print("observation threshold:", opts.observation_time)
print("prediction time:", opts.prediction_time)
print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
print("model save at{}".format(opts.save_dir))
print("===================configuration===================")


DATA_PATH = os.path.join(opts.data_root, opts.dataset)

x_train, y_train, sz_train, vocabulary_size = pickle.load(open(DATA_PATH + 'data_train.pkl', 'rb'))

x_test, y_test, sz_test, _ = pickle.load(open(DATA_PATH + 'data_test.pkl', 'rb'))
x_val, y_val, sz_val, _ = pickle.load(open(DATA_PATH + 'data_val.pkl', 'rb'))
print(len(x_train), len(x_test), len(x_val))
node_vec = pickle.load(open(DATA_PATH + '/node_vec.pkl', 'rb'))

training_iters = opts.training_iters
batch_size = opts.batch_size

display_step = math.ceil(len(x_train) / batch_size)

np.set_printoptions(precision=2)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = DeepCas(opts, sess, node_vec)
saver = tf.train.Saver(max_to_keep=5)
step = 1
best_val_loss = 1000
best_test_loss = 1000
# Keep training until reach max iterations or max_try
train_loss = []
max_try = 10
patience = max_try
start = time.time()
pbar_train = ProgressBar(n_total=math.ceil(training_iters/batch_size), desc='Training')
while step * batch_size < training_iters:

    batch_x, batch_y, batch_sz = get_batch(x_train, y_train, sz_train, step, batch_size=batch_size)
    model.train_batch(batch_x, batch_y, batch_sz)
    train_loss.append(model.get_error(batch_x, batch_y, batch_sz))
    pbar_train(step, {"loss": train_loss[-1]})

    if step % display_step == 0:
        # Calculate batch loss
        val_loss = []
        val_predict = []
        val_truth = []
        pbar_val = ProgressBar(n_total=int(len(y_val) / batch_size), desc='Validating')
        for val_step in range(int(len(y_val) / batch_size)):
            val_x, val_y, val_sz = get_batch(x_val, y_val, sz_val, val_step, batch_size=batch_size)
            val_loss.append(model.get_error(val_x, val_y, val_sz))
            predict_result, truth = model.get_pre_true(val_x, val_y, val_sz)
            predict_result, truth = predict_result.T.squeeze(0), truth.T.squeeze(0)
            val_predict.extend(predict_result)
            val_truth.extend(truth)
            pbar_val(val_step, {'loss': val_loss[-1]})

        test_loss = []
        test_predict = []
        test_truth = []
        pbar_test = ProgressBar(n_total=int(len(y_test) / batch_size), desc='Testing')
        for test_step in range(int(len(y_test) / batch_size)):
            test_x, test_y, test_sz = get_batch(x_test, y_test, sz_test, test_step, batch_size=batch_size)
            test_loss.append(model.get_error(test_x, test_y, test_sz))
            predict_result, truth = model.get_pre_true(test_x, test_y, test_sz)
            predict_result, truth = predict_result.T.squeeze(0), truth.T.squeeze(0)
            test_predict.extend(predict_result)
            test_truth.extend(truth)
            pbar_test(test_step, {'loss': test_loss[-1]})

        if np.mean(val_loss) < best_val_loss:
            saver.save(sess, opts.save_dir, global_step=step)
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try
        val_predict = np.array(val_predict)
        test_predict = np.array(test_predict)
        val_truth = np.array(val_truth)
        test_truth = np.array(test_truth)
        # val metric
        val_mape = mape_loss_func2(val_predict, val_truth)
        val_acc = accuracy(val_predict, val_truth, 0.5)
        val_mSEL = mSEL(MSEL(val_predict, val_truth))
        # test metric
        test_mape = mape_loss_func2(test_predict, test_truth)
        test_acc = accuracy(test_predict, test_truth, 0.5)
        test_mSEL = mSEL(MSEL(test_predict, test_truth))

        print("\n#" + str(int(step / display_step)) +
              ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) + ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) + ", Best Test Loss= " + "{:.6f}".format(best_test_loss))
        print(
            ' Val Loss mSEL:{:5f}, Test Loss mSEL:{:5f},'
            .format(val_mSEL, test_mSEL))
        print(
            ' Val acc:{:5f}, Test acc:{:5f},'
            .format(val_acc, test_acc))
        print(
            ' Val Loss mape:{:5f}, Test Loss mape:{:5f},\n'
            .format(val_mape, test_mape))

        train_loss = []
        patience -= 1
        if not patience:
            break
        print('consuming time in one epoch', time.time() - start)
        start = time.time()

    step += 1
    if step % display_step == 0:
        x_train, y_train, sz_train = shuffle_data(x_train, y_train, sz_train)
        x_val, y_val, sz_val = shuffle_data(x_val, y_val, sz_val)
        x_test, y_test, sz_test = shuffle_data(x_test, y_test, sz_test)
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
