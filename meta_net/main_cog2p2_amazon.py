import sys

sys.path.append('../')
import os.path as osp
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from torch import nn, optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from task_amazon import multitask_data_generator
# from model_cocoop import CoOp
from model_cocoop import CoOp
# from model_node_coop import CoOp
import json
from data_graph import DataHelper
from torch.utils.data import DataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):

    setup_seed(args.seed)
    # gnn = CLIP(args).gnn.to(device)
    clip_model = CLIP(args)  # .to(device)
    # clip_model.load_state_dict(torch.load('../res/amazon/{}/clip.pkl'.format(args.data_name), map_location=device))
    clip_model.load_state_dict(
        torch.load('./res/{}/node_ttgt_8&12_10.pkl', map_location=device))

    # model.load_state_dict(torch.load('../res/cora/h1l1.pkl'))

    task_list, train_idx, val_idx, test_idx = multitask_data_generator(lab_list, labeled_ids, labels, args.k_spt,
                                                                       args.k_val, args.k_qry, args.n_way)

    all_acc = []
    f1_list = []
    # for j in range(len(task_list)):

    train_idx_ts = torch.from_numpy(np.array(train_idx[0])).to(device)
    val_idx_ts = torch.from_numpy(np.array(val_idx[0])).to(device)
    test_idx_ts = torch.from_numpy(np.array(test_idx[0])).to(device)

    train_truth = []
    for a in train_idx[0]:
        train_truth.append(id_lab_dict[str(a)])

    val_truth = []
    for a in val_idx[0]:
        val_truth.append(id_lab_dict[str(a)])

    test_truth = []
    for a in test_idx[0]:
        test_truth.append(id_lab_dict[str(a)])

    task_lables_arr = np.array(labels)[task_list[0]]
    task_labels_dict = dict()
    for i in range(task_lables_arr.shape[0]):
        task_labels_dict[task_lables_arr[i]] = i

    train_truth_ts = [task_labels_dict[train_truth[i]] for i in range(len(train_truth))]
    train_truth_ts = torch.from_numpy(np.array(train_truth_ts)).to(device)

    val_truth_ts = [task_labels_dict[val_truth[i]] for i in range(len(val_truth))]
    val_truth_ts = torch.from_numpy(np.array(val_truth_ts)).to(device)

    test_truth_ts = [task_labels_dict[test_truth[i]] for i in range(len(test_truth))]
    test_truth_ts = torch.from_numpy(np.array(test_truth_ts)).to(device)

    task_lables = task_lables_arr.tolist()
    # print('task_lables', task_lables)
    model = CoOp(args, task_lables, clip_model, device)
    # for param in model.state_dict():
    #     print(param)

    best_val = 0
    patience = 2
    counter = 0
    num_train_samples = train_idx_ts.size(0)
    batch_num = num_train_samples // args.batch_size if num_train_samples % args.batch_size == 0 else num_train_samples // args.batch_size + 1
    epoch_train_orders = np.arange(num_train_samples)

    for epoch in range(1, args.ft_epoch + 1):
        # print('----epoch:' + str(epoch))
        random.shuffle(epoch_train_orders)
        # if epoch % 1 == 0:
            # print('epoch_train_orders: ', epoch_train_orders[:10])

        for i in range(batch_num):
            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, num_train_samples)
            the_idx = epoch_train_orders[start:end]
            model.train()
            train_logits, train_loss = model.forward(train_idx_ts[the_idx], node_f, edge_index, train_truth_ts[the_idx])

            # break

        print("---evaluating---")
        model.eval()
        with torch.no_grad():
            val_loss = 0
            eval_batch_num = num_train_samples // args.eval_batch_size if num_train_samples % args.eval_batch_size == 0 else num_train_samples // args.eval_batch_size + 1
            for i in range(eval_batch_num):
                start = i * args.eval_batch_size
                end = min((i + 1) * args.eval_batch_size, val_idx_ts.size(0))
                res, batch_val_loss = model.forward(val_idx_ts[start:end], node_f, edge_index, val_truth_ts[start:end],
                                                    training=False)
                val_loss += batch_val_loss
            # val_acc = accuracy_score(val_truth_ts.cpu(), res.argmax(dim=1).cpu())
            if val_loss >= best_val:
                counter += 1
                if counter >= patience:
                    break
            else:
                best_val = val_loss
                # torch.save(model, '../res/amazon/{}/g_coop_node.pkl'.format(data_name))
                best_model = model
            counter = 0
    # print('{}th_task_best_val'.format(j), round(best_val, 4))

    # best_model = torch.load('../res/amazon/{}/g_coop_node.pkl'.format(data_name))
    if val_loss >= best_val:
        best_model = model
    print("num of test examples= ", test_idx_ts.size(0))
    best_model.eval()
    with torch.no_grad():
        res_list = []
        test_batch_num = test_idx_ts.size(0) // args.eval_batch_size if test_idx_ts.size(0) % args.eval_batch_size == 0 else test_idx_ts.size(0) // args.eval_batch_size + 1
        for i in range(test_batch_num):
            start = i * args.eval_batch_size
            end = min((i + 1) * args.eval_batch_size, test_idx_ts.size(0))
            batch_res, _ = model.forward(test_idx_ts[start:end], node_f, edge_index, test_truth_ts[start:end],
                                         training=False)
            res_list.append(batch_res)

        # print('res_list', res_list)
        res = torch.cat(res_list, dim=0)
        test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
        # print('{}_task_test_acc'.format(j), round(test_acc, 4))
        all_acc.append(test_acc)
        f1 = f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro')
        f1_list.append(f1)

    ans = round(np.mean(all_acc).item(), 4)
    print('base acc', ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('base macro f1', ans)

    print("\n\n")
    print("----------begin testing new class----------")
    print("\n\n")

    test_idx_ts = torch.from_numpy(np.array(test_idx[1])).to(device)
    test_truth = []
    for a in test_idx[1]:
        test_truth.append(id_lab_dict[str(a)])

    task_lables_arr = np.array(labels)[task_list[1]]
    task_labels_dict = dict()
    for i in range(task_lables_arr.shape[0]):
        task_labels_dict[task_lables_arr[i]] = i

    test_truth_ts = [task_labels_dict[test_truth[i]] for i in range(len(test_truth))]
    test_truth_ts = torch.from_numpy(np.array(test_truth_ts)).to(device)

    test_task_lables = task_lables_arr.tolist()
    print('test_task_lables', test_task_lables[:10])
    test_model = CoOp(args, test_task_lables, clip_model, device)
    # test_model.load_state_dict(best_model.state_dict())
    base_dict = best_model.state_dict()
    # base_dict = model.state_dict()
    new_dict = test_model.state_dict()

    # for param in new_dict:
    #     print(param)
    with torch.no_grad():
        new_dict["model.prompt_learner.ctx"] = base_dict["model.prompt_learner.ctx"]
        new_dict["model.prompt_learner.meta_net.linear1.weight"] = base_dict["model.prompt_learner.meta_net.linear1.weight"]
        new_dict["model.prompt_learner.meta_net.linear1.bias"] = base_dict["model.prompt_learner.meta_net.linear1.bias"]
        new_dict["model.prompt_learner.meta_net.linear2.weight"] = base_dict["model.prompt_learner.meta_net.linear2.weight"]
        new_dict["model.prompt_learner.meta_net.linear2.bias"] = base_dict["model.prompt_learner.meta_net.linear2.bias"]
        test_model.load_state_dict(new_dict)

    test_model.eval()
    with torch.no_grad():
        res_list = []
        test_batch_num = test_idx_ts.size(0) // args.eval_batch_size if test_idx_ts.size(0) % args.eval_batch_size == 0 else test_idx_ts.size(0) // args.eval_batch_size + 1
        for i in range(test_batch_num):
            start = i * args.eval_batch_size
            end = min((i + 1) * args.eval_batch_size, test_idx_ts.size(0))
            batch_res, _ = test_model.forward(test_idx_ts[start:end], node_f, edge_index, test_truth_ts[start:end],
                                         training=False)
            res_list.append(batch_res)

        # print('res_list', res_list)
        res = torch.cat(res_list, dim=0)
        test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
        # print('{}_task_test_acc'.format(j), round(test_acc, 4))
        all_acc.append(test_acc)
        f1 = f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro')
        f1_list.append(f1)

    ans = round(np.mean(all_acc).item(), 4)
    print('new acc', ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('new macro f1', ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ft_epoch', type=int, default=10, help='fine-tune epoch')
    # parser.add_argument('--ft_epoch', type=int, default=1, help='fine-tune epoch')

    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--gnn_input', type=int, default=128)
    # parser.add_argument('--gnn_hid', type=int, default=16)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)

    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    # parser.add_argument('--context_length', type=int, default=120)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.005)
    # parser.add_argument('--prompt_lr', type=float, default=0.001)

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    # parser.add_argument('--class_specific', type=bool, default=True)
    parser.add_argument('--ctx_init', type=bool, default=True)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    # parser.add_argument('--data_name', type=str, default='Arts_Crafts_and_Sewing')
    # parser.add_argument('--data_name', type=str, default='Industrial_and_Scientific')
    parser.add_argument('--data_name', type=str, default="Musical_Instruments")

    args = parser.parse_args()

    # print('args.class_specific=  ', args.class_specific)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")
    print('device', device)
    # data_name = 'cora'
    criterion = nn.BCEWithLogitsLoss()

    # device = torch.device("cpu")
    FType = torch.FloatTensor
    LType = torch.LongTensor

    num_nodes = 0
    tit_dict = json.load(open('./data/amazon/{}_text.json'.format(args.data_name)))
    new_dict = {}
    for i in range(len(tit_dict)):
        new_dict[i] = tit_dict[str(i)]

    id_lab_dict = json.load(open('./data/amazon/{}_id_labels.json'.format(args.data_name)))
    id_lab_list = sorted(id_lab_dict.items(), key=lambda d: int(d[0]))

    labeled_ids = []
    lab_list = []
    for i in id_lab_list:
        # if lab_list[i] == 'nan':
        #     print(lab_list[i])
        if i[1] != 'nan' or i[1] != '' or i[1] != ' ':
            labeled_ids.append(int(i[0]))
            lab_list.append(i[1])

    print('number of labels', len(lab_list))

    edge_index = np.load('./data/amazon/{}_edge.npy'.format(args.data_name))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    # node_f = np.load('../cora/node_f_title.npy').astype(np.float32)
    node_f = np.load('./data/amazon/{}_f_m.npy'.format(args.data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)

    labels = sorted(list(set(lab_list)))

    start = time.perf_counter()
    base_acc_list = []
    base_macf1_list = []

    new_acc_list = []
    new_macf1_list = []

    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
