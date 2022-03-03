import torch
from torch import tensor
import torch.optim as optim
from utils import model
from utils import preprocessing
from config import opts
from tqdm import tqdm
import pickle
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def mape_loss_func2(preds, labels):
    preds = preds.clamp(1).cpu().detach().numpy()
    labels = labels.clamp(1).cpu().detach().numpy()
    return np.fabs((labels-preds)/labels).mean()


def accuracy(preds, labels, bias=0.3):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
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
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return np.square(preds-labels)


if __name__ == "__main__":
    print("===================configuration===================")
    print("learning rate : ", opts.learning_rate)
    print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
    print("observation threshold : ", opts.observation_time)
    print("prediction time : ", opts.prediction_time)
    print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
    print("model save at : {}".format(opts.save_dir))
    print("===================configuration===================")

    # train_examples = pickle.load(open(opts.DATA_PATH + "train.pkl", 'rb'))
    # train_labels = pickle.load(open(opts.DATA_PATH + "train_labels.pkl", 'rb'))

    val_examples = pickle.load(open(opts.DATA_PATH + "val.pkl", 'rb'))
    val_labels = pickle.load(open(opts.DATA_PATH + "val_labels.pkl", 'rb'))

    test_examples = pickle.load(open(opts.DATA_PATH + "test.pkl", 'rb'))
    test_labels = pickle.load(open(opts.DATA_PATH + "test_labels.pkl", 'rb'))

    node_to_index = pickle.load(
        open(opts.DATA_PATH + "node_to_index.pkl", 'rb'))

    topolstm = model.TopoLSTM(data_dir=opts.DATA_PATH,
                              node_index=node_to_index,
                              hidden_size=opts.hidden_size).cuda()
    loss_fn = torch.nn.MSELoss()
    # train_loader = preprocessing.Loader(train_examples,
    #                                     batch_size=opts.batch_size,
    #                                     labels=train_labels,
    #                                     shuffle_data=True)
    val_loader = preprocessing.Loader(val_examples,
                                      batch_size=opts.batch_size,
                                      labels=val_labels,
                                      shuffle_data=True)
    test_loader = preprocessing.Loader(test_examples,
                                       batch_size=opts.batch_size,
                                       labels=test_labels,
                                       shuffle_data=True)

    topolstm.load_state_dict(torch.load(opts.save_dir+'save_{}.pt'.format(opts.observation_time)), strict=True)
    print('==========load finish==========')
    # train_batch_number = len(train_examples) // opts.batch_size + 1
    val_batch_number = len(val_examples) // opts.batch_size + 1
    test_batch_number = len(test_examples) // opts.batch_size + 1
    best_val = 1000
    best_test = 1000
    # for epoch in range(0):
    # train_mean_loss = []
    val_mean_loss = []
    test_mean_loss = []
    # train_acc_loss = []
    val_acc_loss = []
    test_acc_loss = []
    # train_mape_loss = []
    val_mape_loss = []
    test_mape_loss = []
    # train_mSEL_loss = []
    val_mSEL_loss = []
    test_mSEL_loss = []
    for _ in range(val_batch_number):
        topolstm.eval()
        topolstm.zero_grad()
        sequence_matrix, sequence_mask_matrix, topo_mask_matrix, label = val_loader.__call__(
        )
        sequence_matrix = torch.tensor(sequence_matrix,
                                       dtype=torch.long).cuda()
        sequence_mask_matrix = torch.tensor(sequence_mask_matrix,
                                            dtype=torch.float).cuda()
        topo_mask_matrix = torch.tensor(topo_mask_matrix,
                                        dtype=torch.float).cuda()
        label = torch.tensor(label, dtype=torch.float).cuda()
        predict_result = topolstm(sequence_matrix, sequence_mask_matrix,
                                  topo_mask_matrix)
        predict_result = predict_result.squeeze(1)

        # metrics
        loss = loss_fn(predict_result, label)
        val_mSEL_loss.extend(MSEL(predict_result, label))
        val_mean_loss.append(loss.cpu().detach().numpy())
        val_acc_loss.append(accuracy(predict_result, label, 0.5))
        val_mape_loss.append(mape_loss_func2(predict_result, label))
    val_mSEL = mSEL(val_mSEL_loss)
    val_MSEL = np.mean(val_mean_loss)
    val_acc_loss = np.mean(val_acc_loss)
    val_mape = np.mean(val_mape_loss)
    for _ in range(test_batch_number):
        topolstm.eval()
        topolstm.zero_grad()
        sequence_matrix, sequence_mask_matrix, topo_mask_matrix, label = test_loader.__call__(
        )
        sequence_matrix = torch.tensor(sequence_matrix,
                                       dtype=torch.long).cuda()
        sequence_mask_matrix = torch.tensor(sequence_mask_matrix,
                                            dtype=torch.float).cuda()
        topo_mask_matrix = torch.tensor(topo_mask_matrix,
                                        dtype=torch.float).cuda()
        label = torch.tensor(label, dtype=torch.float).cuda()
        predict_result = topolstm(sequence_matrix, sequence_mask_matrix,
                                  topo_mask_matrix)
        predict_result = predict_result.squeeze(1)
        loss = loss_fn(predict_result, label)
        test_mSEL_loss.extend(MSEL(predict_result, label))
        test_mean_loss.append(loss.cpu().detach().numpy())
        test_acc_loss.append(accuracy(predict_result, label, 0.5))
        test_mape_loss.append(mape_loss_func2(predict_result, label))
    test_mSEL = mSEL(test_mSEL_loss)
    test_MSEL = np.mean(test_mean_loss)
    test_acc_loss = np.mean(test_acc_loss)
    test_mape = np.mean(test_mape_loss)
    if best_test > test_MSEL:
        best_test = test_MSEL
    print(
        '# Epochs:{}\nTrain Loss MSEL:{}, Val Loss MSEL:{:5f}, Test Loss MSEL:{:5f},'
        .format(0, None, val_MSEL, test_MSEL))
    print(
        'Train Loss mSEL:{}, Val Loss mSEL:{:5f}, Test Loss mSEL:{:5f},'
        .format(None, val_mSEL, test_mSEL))
    print(
        'Train acc:{}, Val acc:{:5f}, Test acc:{:5f},'
        .format(None, val_acc_loss, test_acc_loss))
    print(
        'Train Loss mape:{}, Val Loss mape:{:5f}, Test Loss mape:{:5f},\n'
        .format(None, val_mape, test_mape))
