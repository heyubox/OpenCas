from sklearn.neural_network import MLPClassifier
from torch_geometric.data import batch
from config import opts
import math
import os
from utils.GNNLSTM import GRU_Cascade_Dynamic_GPN
from torch_geometric.data import Data, Batch
import numpy as np
import torch
import pickle
import time
import sys

from utils.metrics import mape_loss_func2, mSEL, accuracy, MSEL
from utils.utils_func import seed_everything, sparse_mx_to_torch_sparse_tensor, atten_mx, atten_self_mx, gen_ptr, array2tensor, gen_index

sys.path.append('./')


# seed = 8  # even seed is kept still, the results are different
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)


def model_test(model, X_test, Y_test, batch_size, device=None, load_concat=True):
    model.eval()
    graph_info = []
    criterion = torch.nn.MSELoss()
    ptr = []
    outputs_prediction = []
    graph_hidden = []
    rnn_hidden = []
    nodes_F = []
    nodes_map = []
    with torch.no_grad():
        loss_total = []

        Y_preds = []
        Y_trues = []

        # model.hidden = model.init_hidden()
        print('total test iterations: {} '.format(math.ceil(len(X_test) / batch_size)))
        for i in range(math.ceil(len(X_test) / batch_size)):
            train_graph = X_test[batch_size * i:batch_size * (i + 1)]
            targets = Y_test[batch_size * i:batch_size * (i + 1)]
            targets = targets.to(device)
            dval = []
            if len(train_graph) != batch_size:
                continue
            try:
                if load_concat and os.path.exists((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_test.pkl').format(num_try, i, observation_time, interval)):
                    batch_x = pickle.load(open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_test.pkl').format(num_try, i, observation_time, interval), 'rb'))
                    for g in train_graph.reshape(-1):
                        dval.extend(g.id2nodes[k] for k in sorted(g.id2nodes.keys()))
                else:
                    data_list = [Data(array2tensor(g.node_features).to_dense(), g.edge_mat) for g in train_graph.reshape([-1])]

                    batch_x = Batch.from_data_list(data_list)
                    batch_x.ptr = gen_ptr(batch_x.batch)
                    batch_x.ptr = ptr
                    batch_x.atten_edge_index = atten_mx(gen_ptr(batch_x.batch), n_seq, window=5)
                    batch_x.atten_edge_index_self = atten_self_mx(batch_x.ptr)
                    batch_x.scatter_index = None  #
                batch_x.adj = None
                assert batch_x.ptr is not None
                ptr.append(batch_x.ptr.tolist())
                batch_x.to(device)

                outputs, middle_hidden, hidden_step, Aw, middle_feature = model(batch_x, return_node_F=True)
                graph_info.append(Aw)
                outputs_prediction.append(outputs.tolist())
                graph_hidden.append(middle_hidden.tolist())
                rnn_hidden.append(hidden_step.tolist())
                nodes_F.append(middle_feature.cpu())
                nodes_map.append(dict(zip(range(len(dval)), dval)))
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                loss_total.append(loss.item())

                Y_preds.extend(outputs.tolist())
                Y_trues.extend(targets.tolist())

            except Exception as e:
                print(e)

    avg_loss = sum(loss_total) / len(loss_total)
    print('average test loss {:.04f}'.format(avg_loss))
    less_rate = 0.5
    Y_preds_np = np.array(Y_preds)
    Y_trues_np = np.array(Y_trues)

    print("accuracy ( less than {} ): {:.04f}".format(less_rate, accuracy(Y_trues_np, Y_preds_np, less_rate)))
    median_loss = np.median((np.array(Y_preds) - np.array(Y_trues))**2)
    print('median test loss {:.04f}'.format(median_loss))
    graph_hidden, rnn_hidden, ptr = np.array(graph_hidden), np.array(rnn_hidden), np.array(ptr)
    it, bz, step, F = graph_hidden.shape

    return Y_preds_np, Y_trues_np, graph_hidden, rnn_hidden, graph_info, ptr, nodes_F, nodes_map


def model_val(model, X_val, Y_val, batch_size, device=None, load_concat=True):
    model.eval()
    graph_info = []
    criterion = torch.nn.MSELoss()
    ptr = []
    outputs_prediction = []
    graph_hidden = []
    rnn_hidden = []
    nodes_F = []
    nodes_map = []
    with torch.no_grad():
        loss_total = []

        Y_preds = []
        Y_trues = []

        # model.hidden = model.init_hidden()
        print('total val iterations: {} '.format(math.ceil(len(X_val) / batch_size)))
        for i in range(math.ceil(len(X_val) / batch_size)):
            train_graph = X_val[batch_size * i:batch_size * (i + 1)]
            targets = Y_val[batch_size * i:batch_size * (i + 1)]
            targets = targets.to(device)
            dval = []
            if len(train_graph) != batch_size:
                continue
            try:
                if load_concat and os.path.exists((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_val.pkl').format(num_try, i, observation_time, interval)):
                    batch_x = pickle.load(open((opts.DATA_PATH + 'temp{}/batch_x_{}_{}_{}_val.pkl').format(num_try, i, observation_time, interval), 'rb'))
                    for g in train_graph.reshape(-1):
                        dval.extend(g.id2nodes[k] for k in sorted(g.id2nodes.keys()))
                else:
                    data_list = [Data(array2tensor(g.node_features).to_dense(), g.edge_mat) for g in train_graph.reshape([-1])]

                    batch_x = Batch.from_data_list(data_list)
                    batch_x.ptr = gen_ptr(batch_x.batch)
                    batch_x.ptr = ptr
                    batch_x.atten_edge_index = atten_mx(gen_ptr(batch_x.batch), n_seq, window=5)
                    batch_x.atten_edge_index_self = atten_self_mx(batch_x.ptr)
                    batch_x.scatter_index = None  # gen_index(batch_x.ptr, opts.gnn_mlp_hidden)
                    # batch_x.adj = gen_adj(batch_x.x,batch_x.edge_index)
                batch_x.adj = None
                assert batch_x.ptr is not None
                ptr.append(batch_x.ptr.tolist())
                batch_x.to(device)

                outputs, middle_hidden, hidden_step, Aw, middle_feature = model(batch_x, return_node_F=True)
                nodes_F.append(middle_feature.cpu())
                nodes_map.append(dict(zip(range(len(dval)), dval)))
                graph_info.append(Aw)
                outputs_prediction.append(outputs.tolist())
                graph_hidden.append(middle_hidden.tolist())
                rnn_hidden.append(hidden_step.tolist())
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                loss_total.append(loss.item())

                Y_preds.extend(outputs.tolist())
                Y_trues.extend(targets.tolist())

            except Exception as e:
                print(e)

    avg_loss = sum(loss_total) / len(loss_total)
    print('average val loss {:.04f}'.format(avg_loss))
    less_rate = 0.5
    Y_preds_np = np.array(Y_preds)
    Y_trues_np = np.array(Y_trues)
    acc_np = (np.abs(Y_preds_np - Y_trues_np - 1) / (Y_trues_np + 1)) < less_rate
    acc = np.sum(acc_np) / len(acc_np)

    print("accuracy ( less than {} ): {:.04f}".format(less_rate, acc))
    median_loss = np.median((np.array(Y_preds) - np.array(Y_trues))**2)
    print('median val loss {:.04f}'.format(median_loss))
    graph_hidden, rnn_hidden, ptr = np.array(graph_hidden), np.array(rnn_hidden), np.array(ptr)
    it, bz, step, F = graph_hidden.shape

    return Y_preds_np, Y_trues_np, graph_hidden, rnn_hidden, graph_info, ptr, nodes_F, nodes_map


if __name__ == '__main__':
    seed_everything(opts.random_seed)

    num_try = opts.rank
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(num_try)

    batch_size = opts.batch_size

    hidden_size = opts.hidden_size

    observation_time = opts.observation_time
    interval = opts.interval
    file_name = opts.RAWDATA_PATH

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_mlp_layers = opts.num_mlp_layers
    num_layers = opts.num_layers
    # num_layers = 1
    # mlp_hidden = 16
    mlp_hidden = opts.mlp_hidden
    input_features = opts.input_features
    # input_features = 200
    out_features = opts.out_features
    n_seq = interval_num = math.ceil(observation_time / interval)

    middle_size = opts.middle_size

    emb_size = middle_size  # * 2

    load_data = False
    opts.learning_rate = 0.001
    opts.attention_type = 'GAT'
    opts.bidirection = True
    bi = opts.bidirection
    at = opts.attention_type
    gt = opts.gcn_type
    rnn_type = opts.rnn_type = 'None'
    print("===================configuration===================")
    print("learning rate :{} ".format(opts.learning_rate))
    print("num_mlp_layers : {}".format(opts.num_mlp_layers))
    print("num_layers : {}".format(opts.num_layers))
    print("gnn input_features : {}".format(opts.input_features))
    print("gnn output_features : {}".format(opts.gnn_out_features))
    print("input of rnn/transformer : {}".format(emb_size))
    print("hidden_size (out put of rnn/tranformer) : {}".format(opts.hidden_size))
    print("observation hour [{},{}]".format(opts.start_hour, opts.end_hour))
    print("observation interval : {}".format(opts.interval))
    print("observation threshold : {}".format(opts.observation_time))
    print("prediction time : {}".format(opts.prediction_time))
    print("cascade length [{},{}]".format(opts.least_num, opts.up_num))
    print("model save at : {}".format(opts.save_dir))
    print('====bidirected: {}, attention: {}, gcn_type: {}, rnn: {}===='.format(bi, at, gt, rnn_type))
    print("===================configuration===================")

    start = time.time()

    X_valid = pickle.load(open(opts.DATA_PATH + "val.pkl", 'rb'))
    Y_valid = pickle.load(open(opts.DATA_PATH + "val_labels.pkl", 'rb'))

    X_test = pickle.load(open(opts.DATA_PATH + "test.pkl", 'rb'))
    Y_test = pickle.load(open(opts.DATA_PATH + "test_labels.pkl", 'rb'))
    # for gs in X_valid:
    #     g=gs[-1]
    #     if g.current_num > 5:
    #         print('error')
    Y_valid = torch.FloatTensor(Y_valid)
    Y_test = torch.FloatTensor(Y_test)
    print(" len(Y_valid), len(Y_test):",  len(Y_valid), len(Y_test))
    # id_num = pickle.load(open((opts.DATA_PATH + 'idmap_{}.pkl').format(observation_time), 'rb'))
    # id_num = len(id_num)+1
    # print("v_voa:",id_num)
    print('preparing time:{}'.format(time.time() - start))
    model = GRU_Cascade_Dynamic_GPN(n_seq=n_seq,
                                    input_features=input_features,
                                    batch_size=batch_size,
                                    emb_size=emb_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    gnn_out_features=opts.gnn_out_features,
                                    max_len=opts.up_num,
                                    gnn_mlp_hidden=opts.gnn_mlp_hidden,
                                    bidirected=bi,
                                    device=device,
                                    n_vocabulary=None,  # id_num
                                    gcn_type=gt,  # self or geo
                                    atten_type=at,
                                    rnn=rnn_type)
    model.load_state_dict(torch.load(opts.save_dir+"save_{}.pt".format(opts.observation_time)))
    # model.load_state_dict(torch.load(opts.save_dir+"best.pt"))
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=opts.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-10)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters', pytorch_total_params)
    # exit(0)

    import torch.nn.functional as F

    outputs_prediction, outputs_true, graph_hidden, rnn_hidden, graph_info, ptr, nodes_f, nodes_map = model_val(model, X_valid, Y_valid, batch_size, device, True)
    valid_data = {"outputs_prediction": outputs_prediction, "outputs_true": outputs_true, "graph_hidden": graph_hidden,
                  "rnn_hidden": rnn_hidden, "graph_info": graph_info, "ptr": ptr, "nodes_f": nodes_f, 'nodes_map': nodes_map}
    pickle.dump(valid_data, open(opts.DATA_PATH+'valid_ana.pkl', 'wb'))
    outputs_prediction, outputs_true, graph_hidden, rnn_hidden, graph_info, ptr, nodes_f, nodes_map = model_test(model, X_test, Y_test, batch_size, device, True)
    test_data = {"outputs_prediction": outputs_prediction, "outputs_true": outputs_true, "graph_hidden": graph_hidden,
                 "rnn_hidden": rnn_hidden, "graph_info": graph_info, "ptr": ptr, "nodes_f": nodes_f, 'nodes_map': nodes_map}
    pickle.dump(test_data, open(opts.DATA_PATH+'test_ana.pkl', 'wb'))
    # <iter,batch_size,n_seq,hidden>
    print(ptr[:10])
    graph_size = []
    for iter in ptr:
        for idx in range(0, len(iter)-1, n_seq):
            one_batch = iter[idx+1:idx+n_seq+1]-iter[idx:idx+n_seq]
        graph_size.append(one_batch)
    graph_size = np.array(graph_size)
    print(graph_size.shape)
