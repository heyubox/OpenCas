'''
run casflow
'''
from sklearn.utils import shuffle
from config import opts
from utils.tools import *
from tensorflow.keras import backend as K
import pickle
import pickle
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def msle_log2(y_true, y_pred):
    # Calculates the precision
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.) / K.log(2.0)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.) / K.log(2.0)
    return K.mean(K.square(first_log - second_log), axis=-1)


def mape(y_true, y_pred):
    # Calculates the precision

    # first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)/K.log(2.0)
    adjust_y_true = K.clip(y_true, 1.0, None)
    return K.mean(K.abs((y_pred - y_true) / adjust_y_true), axis=-1)

# shuffle to be test after 3600 experiment


def shuffle_data(cascade, global_g, label):
    seed = np.random.randint(low=0, high=100)
    return shuffle(cascade, global_g, label, random_state=seed)


def main(argv=None):

    start_time = time.time()
    print('TF Version:', tf.__version__)

    def casflow_loss(y_true, y_pred):
        mse = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
        node_ce_loss_x = tf.reduce_mean(keras.losses.mean_squared_error(bn_casflow_inputs, node_recon))
        node_kl_loss = -0.5 * tf.reduce_mean(1 + node_z_log_var - tf.square(node_z_mean) - tf.exp(node_z_log_var))

        ce_loss_x = tf.reduce_mean(keras.losses.mean_squared_error(bn_casflow_inputs, recon_x))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        return mse + node_ce_loss_x + ce_loss_x + node_kl_loss + kl_loss - logD_loss

    with open(opts.DATA_PATH + 'train.pkl', 'rb') as ftrain:
        train_cascade, train_global, train_label = pickle.load(ftrain)
    with open(opts.DATA_PATH + 'val.pkl', 'rb') as fval:
        val_cascade, val_global, val_label = pickle.load(fval)
    with open(opts.DATA_PATH + 'test.pkl', 'rb') as ftest:
        test_cascade, test_global, test_label = pickle.load(ftest)

    train_cascade, train_global, train_label = shuffle_data(train_cascade, train_global, train_label)
    val_cascade, val_global, val_label = shuffle_data(val_cascade, val_global, val_label)
    test_cascade, test_global, test_label = shuffle_data(test_cascade, test_global, test_label)

    casflow_inputs = keras.layers.Input(shape=(opts.max_seq, opts.emb_dim))
    bn_casflow_inputs = keras.layers.BatchNormalization()(casflow_inputs)

    vae = VAE(opts.emb_dim, opts.z_dim, opts.max_seq, opts.rnn_units)

    node_z_mean, node_z_log_var = vae.node_encoder(bn_casflow_inputs)
    node_z = Sampling()((node_z_mean, node_z_log_var))
    node_recon = vae.node_decode(node_z)

    z_2 = tf.reshape(node_z, shape=(-1, opts.max_seq, opts.z_dim))

    z_mean, z_log_var = vae.encoder(z_2)
    z = Sampling()((z_mean, z_log_var))

    zk, logD_loss = nf_transformations(z, opts.z_dim, opts.n_flows)

    recon_x = vae.decode(zk)

    gru_1 = keras.layers.Bidirectional(keras.layers.GRU(opts.rnn_units * 2, return_sequences=True))(bn_casflow_inputs)
    gru_2 = keras.layers.Bidirectional(keras.layers.GRU(opts.rnn_units))(gru_1)

    con = keras.layers.concatenate([zk, gru_2])

    mlp_1 = keras.layers.Dense(128, activation='relu')(con)
    mlp_2 = keras.layers.Dense(64, activation='relu')(mlp_1)
    outputs = keras.layers.Dense(1)(mlp_2)

    casflow = keras.Model(inputs=casflow_inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(lr=opts.learning_rate)
    casflow.compile(loss=casflow_loss, optimizer=optimizer, metrics=[mape, msle_log2])
    # print(casflow.summary())
    # return
    train_generator = Generator(train_cascade, train_global, train_label, opts.b_size, opts.max_seq)
    val_generator = Generator(val_cascade, val_global, val_label, opts.b_size, opts.max_seq)
    test_generator = Generator(test_cascade, test_global, test_label, opts.b_size, opts.max_seq)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mape', patience=opts.patience, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint(opts.save_dir+"best_weights.h5", monitor='val_loss',  save_best_only=True, mode='min', save_weights_only=True)

    train_history = casflow.fit_generator(train_generator, validation_data=val_generator, epochs=opts.epochs, verbose=opts.verbose, callbacks=[checkpoint, early_stop])

    print('Training ended!')

    predictions = [1 if pred < 1 else pred for pred in np.squeeze(casflow.predict_generator(test_generator))]
    test_label = [1 if label < 1 else label for label in test_label]
    # test metric
    print('max preds {}, max truth {}'.format(np.max(predictions), np.max(test_label)))
    print('min preds {}, min truth {}'.format(np.min(predictions), np.min(test_label)))
    test_mape = mape_loss_func2(np.log2(predictions), np.log2(test_label))
    test_acc = accuracy(np.log2(predictions), np.log2(test_label), 0.5)
    test_mSEL = mSEL(np.square(np.log2(predictions) - np.log2(test_label)))
    report_msle = np.mean(np.square(np.log2(predictions) - np.log2(test_label)))
    print('test MSEL', report_msle)
    print('test mape', test_mape)
    print('test mSEL', test_mSEL)
    print('test acc', test_acc)

    predictions = [1 if pred < 1 else pred for pred in np.squeeze(casflow.predict_generator(val_generator))]
    val_label = [1 if label < 1 else label for label in val_label]
    print('max preds {}, max truth {}'.format(np.max(predictions), np.max(val_label)))
    print('min preds {}, min truth {}'.format(np.min(predictions), np.min(val_label)))
    val_mape = mape_loss_func2(np.log2(predictions), np.log2(val_label))
    val_acc = accuracy(np.log2(predictions), np.log2(val_label), 0.5)
    val_mSEL = mSEL(np.square(np.log2(predictions) - np.log2(val_label)))

    report_msle = np.mean(np.square(np.log2(predictions) - np.log2(val_label)))
    print('val MSEL', report_msle)
    print('val mape', val_mape)
    print('val mSEL', val_mSEL)
    print('val acc', val_acc)

    print('Finished! Time used: {:.3f}mins.'.format((time.time() - start_time) / 60))


if __name__ == '__main__':
    print("===================configuration===================")
    print("learning rate : ", opts.learning_rate)
    print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
    print("observation threshold : ", opts.observation_time)
    print("prediction time : ", opts.prediction_time)
    print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
    print("model save at : {}".format(opts.save_dir))
    print("===================configuration===================")
    # opts.max_seq = opts.up_num
    main()
