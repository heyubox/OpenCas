import time
from sklearn.utils import shuffle
import numpy as np
import math


def shuffle_data(x, L, y, sz, time_):
    seed = np.random.randint(low=0, high=100)
    return shuffle(x, L, y, sz, time_, random_state=seed)


def get_batch(x, L, y, sz, time, step, opts):
    batch_y = np.zeros(shape=(opts.batch_size, 1))
    batch_x = []
    batch_L = []
    batch_time_interval_index = []
    batch_rnn_index = []
    start = step * opts.batch_size % len(x)
    for i in range(opts.batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        batch_L.append(L[id].todense())
        temp_x = []
        for m in range(len(x[id])):
            temp_x.append(x[id][m].todense())
        batch_x.append(temp_x)
        batch_time_interval_index_sample = []

        for j in range(sz[id]):
            temp_time = np.zeros(shape=(opts.n_time_interval))
            k = int(math.floor(time[id][j] / opts.time_interval))
            temp_time[k] = 1
            batch_time_interval_index_sample.append(temp_time)
        if len(batch_time_interval_index_sample) < opts.n_steps:
            for i in range(opts.n_steps - len(batch_time_interval_index_sample)):
                temp_time_padding = np.zeros(shape=(opts.n_time_interval))
                batch_time_interval_index_sample.append(temp_time_padding)
                i = i + 1
        batch_time_interval_index.append(batch_time_interval_index_sample)
        rnn_index_temp = np.zeros(shape=(opts.n_steps))
        rnn_index_temp[:sz[id]] = 1
        batch_rnn_index.append(rnn_index_temp)

    return batch_x, batch_L, batch_y, batch_time_interval_index, batch_rnn_index


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
