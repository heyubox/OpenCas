import time
import os
import random
import torch
import numpy as np
import json
import pickle
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from .process_cascade import normalize_adj


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    return torch.FloatTensor(np.array(sparse_mx.todense()))


def gen_adj(x, edge_index):
    row, col = edge_index
    import scipy.sparse as sp
    adj = sp.coo_matrix((np.ones(len(row.tolist())), (row.tolist(), col.tolist())), shape=(x.size()[0], x.size()[0]), dtype=np.float32)
    adj = normalize_adj(adj+sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def atten_mx(ptr, n_seq, window=None, self_attend=False):
    edge_index1 = torch.LongTensor([])
    edge_index2 = torch.LongTensor([])
    for i in range(0, len(ptr)-1, n_seq):
        if window:
            assert window % 2 == 1
            w = int((window-1)/2)
            for step in range(n_seq):
                self_g = torch.arange(ptr[i+step], ptr[i+step+1], dtype=torch.int64)
                forward_g = torch.arange(ptr[i+step+1], ptr[i+step+1+w if step+1+w <= n_seq else i+n_seq], dtype=torch.int64)
                backward_g = torch.arange(ptr[i+step-w if step-w >= 0 else 0+i], ptr[i+step], dtype=torch.int64)
                if step == 0:
                    window_g = forward_g
                elif step == n_seq-1:
                    window_g = backward_g
                else:
                    window_g = torch.cat((forward_g, backward_g))
                if self_attend:
                    window_g = torch.cat((window_g, self_g))
                N1 = len(window_g)
                N2 = len(self_g)
                edge_index1 = torch.cat((edge_index1, self_g.repeat_interleave(N1, dim=0)))
                edge_index2 = torch.cat((edge_index2, window_g.repeat(N2)))
        else:
            tmp = torch.arange(ptr[i], ptr[i+n_seq], dtype=torch.int64)
            N = len(tmp)
            edge_index1 = torch.cat((edge_index1, tmp.repeat_interleave(N, dim=0)))
            # print(edge_index1)
            edge_index2 = torch.cat((edge_index2, tmp.repeat(N)))
            # print(tmp.repeat(N))
    return torch.stack((edge_index1, edge_index2))


def atten_self_mx(ptr):
    edge_index1 = torch.LongTensor([])
    edge_index2 = torch.LongTensor([])
    for i in range(len(ptr)-1):
        tmp = torch.arange(ptr[i], ptr[i+1], dtype=torch.int64)
        N = len(tmp)
        edge_index1 = torch.cat((edge_index1, tmp.repeat_interleave(N, dim=0)))
        # print(edge_index1)
        edge_index2 = torch.cat((edge_index2, tmp.repeat(N)))
        # print(tmp.repeat(N))
    return torch.stack((edge_index1, edge_index2))


def gen_ptr(batch):
    '''generate ptr for old version'''
    ptr = [0]
    p = 0
    mark = 0
    for i in batch:
        if i != mark:
            mark = i
            ptr.append(p)
        p += 1
    ptr.append(p)
    return torch.tensor(ptr, dtype=torch.int32)


def array2tensor(node_features):
    import numpy as np
    rows, cols = node_features.nonzero()
    data = node_features.data
    return torch.sparse.LongTensor(torch.LongTensor(np.vstack([rows, cols])), torch.FloatTensor(data), size=(node_features.shape))


def gen_index(ptr, opts):
    index = torch.zeros((int(ptr[-1]), opts.hidden), dtype=torch.int64)
    graph_index = 0
    emb_index = 0
    for graph_index in range(len(ptr) - 1):
        step = 0
        for proj_to in range(ptr[graph_index], ptr[graph_index + 1]):
            index[proj_to][range(opts.hidden)] = emb_index
            emb_index += 1
            step += 1
        emb_index += (opts.up_num - step)
    return index


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
