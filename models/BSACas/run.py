from config import opts
import math
import os
from models.BSACas.utils.GNNTemporal import BSA
import numpy as np
import torch
import pickle
import time
from torch_geometric.data import Data, Batch
from utils.metrics import mape_loss_func2, mSEL, accuracy, MSEL
from utils.utils_func import seed_everything, atten_mx, atten_self_mx, gen_ptr, array2tensor, ProgressBar


def model_train(model, optimizer, X_train, Y_train, batch_size, n_seq=None, device=None, load_concat=False, epoch=0):
    model.train()
    criterion = torch.nn.MSELoss()
    # criterion = sensitive_mse
    loss_total = []
    Y_preds = []
    Y_trues = []
    pbar = ProgressBar(n_total=math.ceil(len(X_train) / batch_size), desc='Training')
    for i in range(math.ceil(len(X_train) / batch_size)):
        optimizer.step()
        model.zero_grad()

        train_graph = X_train[batch_size * i:batch_size * (i + 1)]
        targets = Y_train[batch_size * i:batch_size * (i + 1)]
        if len(train_graph) != batch_size:
            continue

        if not os.path.exists(opts.DATA_PATH + 'temp{}'.format(num_try)):
            os.makedirs(opts.DATA_PATH + 'temp{}'.format(num_try))

        if load_concat and os.path.exists((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}.pkl').format(num_try, i, opts.observation_time, opts.interval)):
            batch_x = pickle.load(open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}.pkl').format(num_try, i, opts.observation_time, opts.interval), 'rb'))
        else:
            data_list = [Data(array2tensor(g.node_features).to_dense(), g.edge_mat) for g in train_graph.reshape([-1])]

            batch_x = Batch.from_data_list(data_list)
            # batch_x.ptr = gen_ptr(batch_x.batch)
            ptr = gen_ptr(batch_x.batch)
            batch_x.atten_edge_index = atten_mx(ptr, n_seq, window=opts.windows_size)
            batch_x.atten_edge_index_self = atten_self_mx(ptr)
            batch_x.scatter_index = None  # gen_index(batch_x.ptr, opts)
            # batch_x.adj = gen_adj(batch_x.x,batch_x.edge_index)
            batch_x.ptr = ptr
            pickle.dump(batch_x, open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}.pkl').format(num_try, i, opts.observation_time, opts.interval), 'wb'))
        batch_x.adj = None  # sparse_mx_to_torch_sparse_tensor(batch_x.adj)

        diff_y = np.array([g.current_num for g in train_graph.reshape([-1])]).reshape(-1, n_seq)
        vec = np.concatenate((np.expand_dims(diff_y[:, 0], 1), diff_y[:, :-1]), axis=1)
        diff_target = torch.Tensor(diff_y-vec).reshape(-1)  # [b*n_seq]
        diff_target = diff_target.to(device)

        batch_x.to(device)

        outputs, hidden, _, _, diff_out = model(batch_x)

        outputs = outputs.squeeze()
        targets = targets.to(device)

        loss_ = criterion(outputs, targets)
        loss_diff = criterion(diff_out, diff_target)
        if epoch < 10:
            # loss = 0.005*loss_+loss_diff
            loss = loss_
        else:
            # loss = 0.001*loss_+loss_diff
            loss = loss_

        loss_total.append(loss_.item())
        pbar(i, {'loss': loss.item()})
        Y_preds.extend(outputs.tolist())
        Y_trues.extend(targets.tolist())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
    avg_loss = sum(loss_total) / len(loss_total)
    print('average train loss {:.04f}'.format(avg_loss))
    return avg_loss


def model_test(model, X_test, Y_test, batch_size, n_seq=None, device=None, load_concat=False):
    model.eval()

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        loss_total = []

        Y_preds = []
        Y_trues = []

        # model.hidden = model.init_hidden()
        pbar = ProgressBar(n_total=math.ceil(len(X_test) / batch_size), desc='Testing')
        for i in range(math.ceil(len(X_test) / batch_size)):
            train_graph = X_test[batch_size * i:batch_size * (i + 1)]
            targets = Y_test[batch_size * i:batch_size * (i + 1)]
            targets = targets.to(device)
            if len(train_graph) != batch_size:
                continue
            try:
                if load_concat and os.path.exists((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_test.pkl').format(num_try, i, opts.observation_time, opts.interval)):
                    batch_x = pickle.load(open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_test.pkl').format(num_try, i, opts.observation_time, opts.interval), 'rb'))
                else:
                    data_list = [Data(array2tensor(g.node_features).to_dense(), g.edge_mat) for g in train_graph.reshape([-1])]

                    batch_x = Batch.from_data_list(data_list)
                    # batch_x.ptr = gen_ptr(batch_x.batch)
                    ptr = gen_ptr(batch_x.batch)
                    batch_x.atten_edge_index = atten_mx(ptr, n_seq, window=opts.windows_size)
                    batch_x.atten_edge_index_self = atten_self_mx(ptr)
                    batch_x.scatter_index = None  # gen_index(batch_x.ptr, opts)
                    # batch_x.adj = gen_adj(batch_x.x,batch_x.edge_index)
                    batch_x.ptr = ptr
                    pickle.dump(batch_x, open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_test.pkl').format(num_try, i, opts.observation_time, opts.interval), 'wb'))
                batch_x.adj = None  # sparse_mx_to_torch_sparse_tensor(batch_x.adj)
                # batch_x.ptr = None

                batch_x.to(device)

                outputs, hidden, _, _, _ = model(batch_x)
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                loss_total.append(loss.item())
                pbar(i, {'loss': loss.item()})
                Y_preds.extend(outputs.tolist())
                Y_trues.extend(targets.tolist())

            except Exception as e:
                print(e)

    avg_loss = sum(loss_total) / len(loss_total)
    print('average test loss {:.04f}'.format(avg_loss))
    Y_preds_np = np.array(Y_preds)
    Y_trues_np = np.array(Y_trues)

    # test metric
    test_mape = mape_loss_func2(Y_preds_np, Y_trues_np)
    test_acc = accuracy(Y_preds_np, Y_trues_np, 0.5)
    test_mSEL = mSEL(MSEL(Y_preds_np, Y_trues_np))

    print(
        'Test Loss mSEL:{:5f},'
        .format(test_mSEL))
    print(
        'Test acc:{:5f},'
        .format(test_acc))
    print(
        'Test Loss mape:{:5f},'
        .format(test_mape))

    return avg_loss, test_mSEL


def model_val(model, X_val, Y_val, batch_size, n_seq=None, device=None, load_concat=False):
    model.eval()

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        loss_total = []

        Y_preds = []
        Y_trues = []

        # model.hidden = model.init_hidden()
        pbar = ProgressBar(n_total=math.ceil(len(X_val) / batch_size), desc='Validating')
        for i in range(math.ceil(len(X_val) / batch_size)):
            train_graph = X_val[batch_size * i:batch_size * (i + 1)]
            targets = Y_val[batch_size * i:batch_size * (i + 1)]
            targets = targets.to(device)
            if len(train_graph) != batch_size:
                continue
            try:
                if load_concat and os.path.exists((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_val.pkl').format(num_try, i, opts.observation_time, opts.interval)):
                    batch_x = pickle.load(open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_val.pkl').format(num_try, i, opts.observation_time, opts.interval), 'rb'))
                else:
                    data_list = [Data(array2tensor(g.node_features).to_dense(), g.edge_mat) for g in train_graph.reshape([-1])]

                    batch_x = Batch.from_data_list(data_list)
                    # batch_x.ptr = gen_ptr(batch_x.batch)
                    ptr = gen_ptr(batch_x.batch)
                    batch_x.ptr = ptr
                    batch_x.atten_edge_index = atten_mx(ptr, n_seq, window=opts.windows_size)
                    batch_x.atten_edge_index_self = atten_self_mx(ptr)
                    batch_x.scatter_index = None  # gen_index(batch_x.ptr, opts)
                    # batch_x.adj = gen_adj(batch_x.x,batch_x.edge_index)

                    pickle.dump(batch_x, open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_val.pkl').format(num_try, i, opts.observation_time, opts.interval), 'wb'))
                batch_x.adj = None  # sparse_mx_to_torch_sparse_tensor(batch_x.adj)

                batch_x.to(device)

                outputs, hidden, _, _, _ = model(batch_x)
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                loss_total.append(loss.item())
                pbar(i, {'loss': loss.item()})
                Y_preds.extend(outputs.tolist())
                Y_trues.extend(targets.tolist())

            except Exception as e:
                print(e)

    avg_loss = sum(loss_total) / len(loss_total)
    print('average val loss {:.04f}'.format(avg_loss))
    Y_preds_np = np.array(Y_preds)
    Y_trues_np = np.array(Y_trues)
    # val metric
    val_mape = mape_loss_func2(Y_preds_np, Y_trues_np)
    val_acc = accuracy(Y_preds_np, Y_trues_np, 0.5)
    val_mSEL = mSEL(MSEL(Y_preds_np, Y_trues_np))

    print(
        ' Val Loss mSEL:{:5f}'
        .format(val_mSEL))
    print(
        ' Val acc:{:5f}'
        .format(val_acc))
    print(
        ' Val Loss mape:{:5f}'
        .format(val_mape))

    return avg_loss, val_mSEL


if __name__ == '__main__':
    num_try = opts.rank
    torch.cuda.set_device(opts.rank)
    seed_everything(opts.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # opts.batch_size=1
    opts.n_seq = opts.interval_num = math.ceil(opts.observation_time / opts.interval)

    print("===================configuration===================")
    print("learning rate :{} ".format(opts.learning_rate))
    print("num_mlp_layers : {}".format(opts.num_mlp_layers))
    print("num_layers : {}".format(opts.num_layers))
    print("gnn input_features : {}".format(opts.input_features))
    print("gnn output_features : {}".format(opts.gnn_out_features))
    print("input of rnn/transformer : {}".format(opts.middle_size))
    print("hidden_size (out put of rnn/tranformer) : {}".format(opts.hidden_size))
    print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
    print("observation interval : {}".format(opts.interval))
    print("observation threshold : {}".format(opts.observation_time))
    print("prediction time : {}".format(opts.prediction_time))
    print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
    print("model save at : {}".format(opts.save_dir))
    print('====bidirected: {}, attention: {}, rnn: {}===='.format(opts.bidirection, opts.attention_type, opts.rnn_type))
    print("===================configuration===================")

    start = time.time()
    X_train = pickle.load(open(opts.DATA_PATH + "train.pkl", 'rb'))
    Y_train = pickle.load(open(opts.DATA_PATH + "train_labels.pkl", 'rb'))

    X_valid = pickle.load(open(opts.DATA_PATH + "val.pkl", 'rb'))
    Y_valid = pickle.load(open(opts.DATA_PATH + "val_labels.pkl", 'rb'))

    X_test = pickle.load(open(opts.DATA_PATH + "test.pkl", 'rb'))
    Y_test = pickle.load(open(opts.DATA_PATH + "test_labels.pkl", 'rb'))

    Y_train = torch.FloatTensor(Y_train)
    Y_valid = torch.FloatTensor(Y_valid)
    Y_test = torch.FloatTensor(Y_test)
    print("len(Y_train), len(Y_valid), len(Y_test):", len(Y_train), len(Y_valid), len(Y_test))
    print("train: max, min: ", torch.max(Y_train).numpy(), torch.min(Y_train).numpy())
    print("val: max, min: ", torch.max(Y_valid).numpy(), torch.min(Y_valid).numpy())
    print("test: max, min: ", torch.max(Y_test).numpy(), torch.min(Y_test).numpy())
    print('preparing time:{}'.format(time.time() - start))

    model = BSA(n_seq=opts.n_seq,
                input_features=opts.input_features,
                batch_size=opts.batch_size,
                emb_size=opts.middle_size,
                hidden_size=opts.hidden_size,
                num_layers=opts.num_layers,
                gnn_out_features=opts.gnn_out_features,
                max_len=opts.up_num,
                gnn_mlp_hidden=opts.gnn_mlp_hidden,
                bidirected=opts.bidirection,
                device=device,
                atten_type=opts.attention_type,
                rnn=opts.rnn_type)
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=opts.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=1e-5)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-10)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters', pytorch_total_params)
    val_loss_list = []
    test_loss_list = []
    train_los_list = []
    epochs = opts.epochs
    best_test_loss = 1000
    best_median_loss = 1000
    for epoch in range(epochs):
        print('---------------*****---------------')
        print('Starting training the epoch: {} '.format(epoch))

        start = time.time()
        load_concat = False
        if epoch >= 1:
            load_concat = True

        if epoch == 20:
            optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, weight_decay=1e-8, momentum=0.9)

        train_loss = model_train(model, optimizer, X_train, Y_train, opts.batch_size, opts.n_seq, device, load_concat, epoch)
        print('Train Consuming time:{:.2f} in the epoch: {}'.format(time.time() - start, epoch))

        v_avg_loss, v_median_loss = model_val(model, X_valid, Y_valid, opts.batch_size, opts.n_seq, device, load_concat)
        t_avg_loss, t_median_loss = model_test(model, X_test, Y_test, opts.batch_size, opts.n_seq, device, load_concat)
        train_los_list.append(train_loss)
        test_loss_list.append(t_avg_loss)
        val_loss_list.append(v_avg_loss)
        if t_avg_loss < best_test_loss:
            torch.save(model.state_dict(), opts.save_dir+'save_{}.pt'.format(opts.observation_time))
            best_test_loss = t_avg_loss

        if t_median_loss < best_median_loss:
            best_median_loss = t_median_loss
        print('Consuming time:{:.2f} in the epoch: {}'.format(time.time() - start, epoch))
        print('Best test loss is {:.2f}'.format(best_test_loss))
        if v_avg_loss <= 0.01:
            break

        if scheduler:
            scheduler.step()

        if scheduler:
            print("epoch:{} current learning rate :{}".format(epoch, scheduler.get_lr()))
