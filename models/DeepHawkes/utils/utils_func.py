import numpy as np
from sklearn.utils import shuffle
import math
import time


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


def get_batch(x, y, sz, time, rnn_index, step, opts):
    batch_y = np.zeros(shape=(opts.batch_size, 1))
    batch_x = []
    batch_x_indict = []
    batch_time_interval_index = []
    batch_rnn_index = []
    start = step * opts.batch_size % len(x)

    # print start
    for i in range(opts.batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        for j in range(sz[id]):
            batch_x.append(x[id][j])
            # time_interval
            temp_time = np.zeros(shape=(opts.n_time_interval))
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


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
