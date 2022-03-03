import math
from config import opts
import time
import gzip
import six.moves.cPickle as pickle
from sklearn.utils import shuffle
from utils.model import DeepCas
import tensorflow as tf
import os
import numpy as np
import sys

sys.path.append('./')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


tf.set_random_seed(0)

tf.flags.DEFINE_float("learning_rate", opts.learning_rate, "learning_rate.")
tf.flags.DEFINE_float("emb_learning_rate", opts.emb_learning_rate, "embedding learning_rate.")
# tf.flags.DEFINE_float("emb_learning_rate", opts.emb_learning_rate, "embedding learning_rate.")
# tf.flags.DEFINE_float("learning_rate", opts.learning_rate, "learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", opts.sequence_batch_size, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size", opts.batch_size, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", opts.n_hidden_gru, "hidden gru size.")
tf.flags.DEFINE_float("l1", opts.l1, "l1.")
tf.flags.DEFINE_float("l2", opts.l2, "l2.")
tf.flags.DEFINE_float("l1l2", opts.l1l2, "l1l2.")
tf.flags.DEFINE_string("activation", opts.activation, "activation function.")
tf.flags.DEFINE_integer("n_sequences", opts.n_sequences, "num of sequences.")
tf.flags.DEFINE_integer("training_iters", opts.training_iters, "max training iters.")
tf.flags.DEFINE_integer("display_step", opts.display_step, "display step.")
# 注意这里加了一个degree的信息，所以维度+1
# tf.flags.DEFINE_integer("embedding_size", opts.embedding_size, "embedding size.")
# tf.flags.DEFINE_integer("n_input", opts.n_input, "input size.")
# tf.flags.DEFINE_integer("embedding_size", opts.embedding_size, "embedding size.")
# tf.flags.DEFINE_integer("n_input", opts.n_input, "input size.")
#
tf.flags.DEFINE_integer("embedding_size", opts.embedding_size, "embedding size.")
tf.flags.DEFINE_integer("n_input", opts.n_input, "input size.")

tf.flags.DEFINE_integer("n_steps", opts.n_steps, "num of step.")
tf.flags.DEFINE_integer("n_hidden_dense1", opts.n_hidden_dense1, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", opts.n_hidden_dense2, "dense2 size.")
tf.flags.DEFINE_string("version", opts.version, "data version.")
# tf.flags.DEFINE_integer("max_grad_norm", opts.max_grad_norm, "gradient clip.")
tf.flags.DEFINE_integer("max_grad_norm", opts.max_grad_norm, "gradient clip.")
tf.flags.DEFINE_float("stddev", opts.stddev, "initialization stddev.")

tf.flags.DEFINE_float("dropout_prob", opts.dropout_prob, "dropout probability.")

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

print("===================configuration===================")
print("dropout prob:", config.dropout_prob)
print("l2", config.l2)
print("learning rate:", config.learning_rate)
print("emb_learning_rate:", config.emb_learning_rate)
print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
print("observation threshold:", opts.observation_time)
print("prediction time:", opts.prediction_time)
print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
print("model save at{}".format(opts.save_dir))
print("===================configuration===================")


def mape_loss_func2(preds, labels):
    preds = np.clip(np.array(preds), 1, 1000)
    labels = np.clip(np.array(labels), 1, 1000)
    return np.fabs((labels-preds)/labels).mean()


def accuracy(preds, labels, bias=0.5):
    preds, labels = np.array(preds), np.array(labels)
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


def get_batch(x, y, sz, step, batch_size=128):
    batch_x = np.zeros((batch_size, len(x[0]), len(x[0][0])))
    batch_y = np.zeros((batch_size, 1))
    batch_sz = np.zeros((batch_size, 1))
    start = step * batch_size % len(x)
    for i in range(batch_size):
        batch_y[i, 0] = y[(i + start) % len(x)]
        batch_sz[i, 0] = sz[(i + start) % len(x)]
        batch_x[i, :] = np.array(x[(i + start) % len(x)])
    return batch_x, batch_y, batch_sz


version = config.version

DATA_PATH = os.path.join(opts.data_root, opts.dataset)


x_test, y_test, sz_test, _ = pickle.load(open(DATA_PATH + 'data_test.pkl', 'rb'))
x_val, y_val, sz_val, _ = pickle.load(open(DATA_PATH + 'data_val.pkl', 'rb'))
node_vec = pickle.load(open(DATA_PATH + '/node_vec.pkl', 'rb'))

training_iters = config.training_iters
batch_size = config.batch_size


# print(display_step)
# display_step = 1
model_save_path = opts.save_dir

np.set_printoptions(precision=2)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = DeepCas(config, sess, node_vec)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(model_save_path)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("\n==============load success====================\n")
    print(model_save_path)
step = 1
best_val_loss = 1000
best_test_loss = 1000
# Keep training until reach max iterations or max_try
train_loss = []
max_try = 30
patience = max_try


# Calculate batch loss
val_loss = []
val_predict = []
val_truth = []
for val_step in range(int(len(y_val) / batch_size)):
    val_x, val_y, val_sz = get_batch(x_val, y_val, sz_val, val_step, batch_size=batch_size)
    val_loss.append(model.get_error(val_x, val_y, val_sz))
    predict_result, truth = model.get_pre_true(val_x, val_y, val_sz)
    predict_result, truth = predict_result.T.squeeze(0), truth.T.squeeze(0)
    val_predict.extend(predict_result)
    val_truth.extend(truth)

test_loss = []
test_predict = []
test_truth = []
for test_step in range(int(len(y_test) / batch_size)):
    test_x, test_y, test_sz = get_batch(x_test, y_test, sz_test, test_step, batch_size=batch_size)
    test_loss.append(model.get_error(test_x, test_y, test_sz))
    predict_result, truth = model.get_pre_true(test_x, test_y, test_sz)
    predict_result, truth = predict_result.T.squeeze(0), truth.T.squeeze(0)
    test_predict.extend(predict_result)
    test_truth.extend(truth)
if np.mean(val_loss) < best_val_loss:
    saver.save(sess, opts.save_dir, global_step=step)
    best_val_loss = np.mean(val_loss)
    best_test_loss = np.mean(test_loss)
    patience = max_try

# val metric
val_mape = mape_loss_func2(val_predict, val_truth)
val_acc = accuracy(val_predict, val_truth)
val_mSEL = mSEL(MSEL(val_predict, val_truth))
# test metric
test_mape = mape_loss_func2(test_predict, test_truth)
test_acc = accuracy(test_predict, test_truth)
test_mSEL = mSEL(MSEL(test_predict, test_truth))

print("#" +
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


print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
