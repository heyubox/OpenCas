import numpy as np


def mape_loss_func2(preds, labels):
    preds = np.clip(np.array(preds), 1, 1000)
    labels = np.clip(np.array(labels), 1, 1000)
    return np.fabs((labels-preds)/labels).mean()


def accuracy(preds, labels, bias=0.5):
    # preds += 0.01
    # labels += 0.01
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
