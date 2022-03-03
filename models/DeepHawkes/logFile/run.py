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


n_nodes, n_sequences, n_steps = pickle.load(open(opts.information, 'rb'))
opts.n_sequences = n_sequences
opts.n_steps = n_steps
print("dataset information: nodes:{}, n_sequence:{}, n_steps:{} ".format(n_nodes, n_sequences, n_steps))
# tf.flags.DEFINE_integer("n_sequences", n_sequences, "num of sequences.")
# tf.flags.DEFINE_integer("n_steps", n_steps, "num of step.")
# tf.flags.DEFINE_integer("time_interval", opts.time_interval, "the time interval")
# tf.flags.DEFINE_integer("n_time_interval", opts.n_time_interval, "the number of  time interval")
# tf.flags.DEFINE_float("learning_rate", opts.learning_rate, "learning_rate.")
# tf.flags.DEFINE_integer("sequence_batch_size", opts.sequence_batch_size, "sequence batch size.")
# tf.flags.DEFINE_integer("batch_size", opts.batch_size, "batch size.")
# tf.flags.DEFINE_integer("n_hidden_gru", opts.n_hidden_gru, "hidden gru size.")
# tf.flags.DEFINE_float("l1", opts.l1, "l1.")
# tf.flags.DEFINE_float("l2", opts.l2, "l2.")
# tf.flags.DEFINE_float("l1l2", opts.l1l2, "l1l2.")
# tf.flags.DEFINE_string("activation", opts.activation, "activation function.")
# tf.flags.DEFINE_integer("training_iters", opts.training_iters, "max training iters.")
# tf.flags.DEFINE_integer("display_step", opts.display_step, "display step.")
# tf.flags.DEFINE_integer("embedding_size", opts.embedding_size, "embedding size.")
# tf.flags.DEFINE_integer("n_input", opts.n_input, "input size.")
# tf.flags.DEFINE_integer("n_hidden_dense1", opts.n_hidden_dense1, "dense1 size.")
# tf.flags.DEFINE_integer("n_hidden_dense2", opts.n_hidden_dense2, "dense2 size.")
# tf.flags.DEFINE_string("version", opts.version, "data version.")
# tf.flags.DEFINE_integer("max_grad_norm", opts.max_grad_norm, "gradient clip.")
# tf.flags.DEFINE_float("stddev", opts.stddev, "initialization stddev.")
# tf.flags.DEFINE_float("emb_learning_rate", opts.emb_learning_rate, "embedding learning_rate.")
# tf.flags.DEFINE_float("dropout_prob", opts.dropout, "dropout probability.")
# tf.flags.DEFINE_boolean("PRETRAIN", opts.PRETRAIN, "Loading PRETRAIN models or not.")
# tf.flags.DEFINE_boolean("fix", opts.fix, "Fix the pretrained embedding or not.")
# tf.flags.DEFINE_boolean("classification", opts.classification, "classification or regression.")
# tf.flags.DEFINE_integer("n_class", opts.n_class, "number of class if do classification.")
# tf.flags.DEFINE_boolean("one_dense_layer", opts.one_dense_layer, "number of dense layer out output.")
# tf.flags.DEFINE_string('rawdataset', 'value', 'The explanation of this parameter is ing')
# tf.flags.DEFINE_string('dataset', 'value', 'The explanation of this parameter is ing')
# tf.flags.DEFINE_string('observation_time', 'value', 'The explanation of this parameter is ing')
# tf.flags.DEFINE_string('interval', 'value', 'The explanation of this parameter is ing')
# tf.flags.DEFINE_string('least_num', 'value', 'The explanation of this parameter is ing')
# tf.flags.DEFINE_string('up_num', 'value', 'The explanation of this parameter is ing')

# config = tf.flags.FLAGS
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


def mape_loss_func2(preds, labels):
    preds = np.clip(np.array(preds), 1, 1000)
    labels = np.clip(np.array(labels), 1, 1000)
    return np.fabs((labels-preds)/labels).mean()


def accuracy(preds, labels, bias=0.5):
    preds += 0.1
    labels += 0.1
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
            k = int(math.floor(time[id][j] / opts.time_interval))
            # in observation_num model, the k can be larger than n_time_interval
            if k >= opts.n_time_interval:
                k = opts.n_time_interval - 1

            temp_time[k] = 1
            batch_time_interval_index.append(temp_time)

            # rnn index
            temp_rnn = np.zeros(shape=(opts.n_steps))
            if rnn_index[id][j] - 1 >= 0:
                temp_rnn[rnn_index[id][j] - 1] = 1
            batch_rnn_index.append(temp_rnn)

            for k in range(2 * opts.n_hidden_gru):
                batch_x_indict.append([i, j, k])

    if opts.classification:
        batch_y = get_y_label(batch_y)

    return batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index


x_train, y_train, sz_train, time_train, rnn_index_train, vocabulary_size = pickle.load(open(opts.train_pkl, 'rb'))
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


analysis_data(y_train, y_test, y_val)
print(len(x_train), len(x_test), len(x_val))

training_iters = opts.training_iters
batch_size = opts.batch_size
# display_step = min(opts.display_step, len(sz_train) / batch_size)
# display_step = 5
display_step = math.ceil(len(x_train) / batch_size)  # display for each epoch
epoch_batch = math.ceil(len(sz_train) / batch_size)

# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = SDPP(opts, sess, n_nodes)
# sess.graph.finalize()
# sess.run(tf.global_variables_initializer())

step = 0
best_val_loss = 1000
best_test_loss = 1000

# train_writer = tf.summary.FileWriter("../model_save/deephawkes/train", sess.graph)

# Keep training until reach max iterations or max_try
train_loss = []
train_mse = []
max_try = 8
patience = max_try

start = time.time()
# 保存模型
saver = tf.train.Saver(max_to_keep=5)
pbar = tqdm(total=training_iters)
while step * batch_size < training_iters*2:
    batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index = get_batch(x_train,
                                                                                             y_train,
                                                                                             sz_train,
                                                                                             time_train,
                                                                                             rnn_index_train,
                                                                                             opts.n_time_interval,
                                                                                             step,
                                                                                             batch_size=batch_size)
    pred = model.train_batch(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)
    # saver = tf.train.Saver()
    # saver.save(model.sess, '../save_models/train_model_%s.ckpt'%step)
    # train_writer.add_summary(summary, step)
    # preds = model.predict(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)

    loss_train = model.get_error(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)
    train_loss.append(loss_train)

    # if (step + 1) % display_step == 0:
    #     saver.save(sess, opts.save_dir, global_step=step)

    if (step + 1) % display_step == 0:
        print('finish one epoch, time{}'.format(time.time() - start))

    if (step + 1) % display_step == 0:
        # for train_step in range(int(len(y_train) / batch_size)):
        #     val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index = get_batch(x_train,
        #                                                                                    y_train,
        #                                                                                    sz_train,
        #                                                                                    time_train,
        #                                                                                    rnn_index_train,
        #                                                                                    opts.n_time_interval,
        #                                                                                    train_step,
        #                                                                                    batch_size=batch_size)
        #     train_loss.append(model.get_error(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index))

        #     predictions = model.predict(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index)
        #     if opts.classification:
        #         predictions = predictions.argmax(axis=-1)
        #     train_pred.extend(predictions.squeeze().tolist())
        #     train_truth.extend(val_y.squeeze().tolist())

        # # print('train acc: ', np.sum(np.equal(np.array(train_pred), np.array(train_truth))) / len(train_pred))
        # print('\ntrain mseloss :', np.sum((np.array(train_pred) - np.array(train_truth))**2) / len(train_pred))
        # print('lambda1', lambda1)
        val_loss = []
        val_pred, val_truth = [], []
        for val_step in range(int(len(y_val) / batch_size)):
            val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index = get_batch(x_val, y_val, sz_val, time_val, rnn_index_val, opts.n_time_interval, val_step, batch_size=batch_size)
            val_loss.append(model.get_error(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index))

            predictions = model.predict(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index)
            if opts.classification:
                predictions = predictions.argmax(axis=-1)
            val_pred.extend(predictions.squeeze().tolist())
            val_truth.extend(val_y.squeeze().tolist())

        test_loss = []
        test_mse = []
        test_pred, test_truth = [], []
        for test_step in range(int(len(y_test) / batch_size)):
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_test,
                                                                                                y_test,
                                                                                                sz_test,
                                                                                                time_test,
                                                                                                rnn_index_test,
                                                                                                opts.n_time_interval,
                                                                                                test_step,
                                                                                                batch_size=batch_size)
            test_loss.append(model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
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
            print("#" + str(int(step / display_step)) + ", Training Loss= " + "{:.5f}".format(np.mean(train_loss)) + ", Valid Loss= " + "{:.5f}".format(np.mean(val_loss)) + ", Valid median Loss= " +
                  "{:.5f}".format(median_val_loss) + ", Best Valid Loss= " + "{:.5f}".format(best_val_loss) + ", Test Loss= " + "{:.5f}".format(np.mean(test_loss)) + ", Test median  Loss= " +
                  "{:.5f}".format(median_loss) + ", Best Test Loss= " + "{:.5f}".format(best_test_loss))
            print(' Val Loss mSEL:{:5f}, Test Loss mSEL:{:5f},'.format(val_mSEL, test_mSEL))
            print(' Val acc:{:5f}, Test acc:{:5f},'.format(val_acc, test_acc))
            print(' Val Loss mape:{:5f}, Test Loss mape:{:5f},\n'.format(val_mape, test_mape))
        else:
            print("#" + str(int(step / display_step)) + ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + ", Valid acc= " +
                  "{:.6f}".format(acc_valid) + ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) + ", Test acc= " + "{:.6f}".format(acc_test) + ", Best Valid Loss= " +
                  "{:.6f}".format(best_val_loss) + ", Best Test Loss= " + "{:.6f}".format(best_test_loss))

        # print("#" + str(step / display_step) +
        #       ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) +
        #       ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) +
        #       ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
        #       ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) +
        #       ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
        #       )
        train_loss = []
        patience -= 1
        if not patience:
            break
        # if best_val_loss <2.35:
    #    break
    step += 1
    pbar.update(batch_size)
    if step % epoch_batch == 0:
        x_train, y_train, sz_train, time_train, rnn_index_train = shuffle_data(x_train, y_train, sz_train, time_train, rnn_index_train)
        x_test, y_test, sz_test, time_test, rnn_index_test = shuffle_data(x_test, y_test, sz_test, time_test, rnn_index_test)
        x_val, y_val, sz_val, time_val, rnn_index_val, = shuffle_data(x_val, y_val, sz_val, time_val, rnn_index_val)

# saver.restore(sess, '../checkpoints/hawkes/hawkes.ckpt')

# print len(predict_result),len(y_test)
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
