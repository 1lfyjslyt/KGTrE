import os
import pickle

import torch
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score, accuracy_score
from .load_base import load_kg
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np

def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


class AGRE(nn.Module):


    def __init__(self, args, n_entity, n_relation):

        super(AGRE, self).__init__()
        self.dim = args.dim
        self.p = args.p
        self.n_relation = n_relation
        self.path_len = args.path_len

        a = np.concatenate((args.embeddings['P'], args.embeddings['V']), axis=0)
        b = np.concatenate((args.embeddings['C'], args.embeddings['A']), axis=0)
        entity_embedding_matrix = np.concatenate((a, b), axis=0)
        entity_embedding_matrix = torch.tensor(entity_embedding_matrix)
        relation_embedding_matrix = np.concatenate((args.relation, t.randn(n_relation+1 - len(args.relation), args.dim)), axis=0)
        relation_embedding_matrix = torch.tensor(relation_embedding_matrix)

        self.entity_embedding_matrix = nn.Parameter(entity_embedding_matrix)
        self.relation_embedding_matrix = nn.Parameter(relation_embedding_matrix)

        self.rnn = nn.RNN(2 * args.dim, 2 * args.dim)
        # self.rnn = nn.LSTM(2 * args.dim, args.dim)
        self.weight_predict = nn.Linear(2 * args.dim, 1)
        self.weight_path = nn.Parameter(t.randn(args.p, args.p))
        self.weight_attention = nn.Linear(2 * args.dim, 1)

    def forward(self, paths_list, relation_dict):
        embeddings_list = self.get_embedding(paths_list, relation_dict)
        embeddings = t.cat(embeddings_list, dim=0)
        h = self.rnn(embeddings)[0][-1]
        h = h.reshape(-1, self.p, 2 * self.dim)
        h = t.sigmoid(t.matmul(self.weight_path, h))

        # attention
        attention = t.sigmoid(self.weight_attention(h))
        attention = t.softmax(attention, dim=1)
        final_hidden_states = (attention * h).sum(dim=1)
        predicts = t.sigmoid(self.weight_predict(final_hidden_states).reshape(-1))
        return predicts

    def get_embedding(self, paths_list, relation_dict):

        embeddings_list = []
        zeros = t.zeros(self.p, self.dim)
        if t.cuda.is_available():
            zeros = zeros.to(self.entity_embedding_matrix.data.device)

        for i in range(self.path_len+1):
            i_entity_embedding_list = []
            i_relation_embedding_list = []
            for paths in paths_list:

                if len(paths) == 0:
                    i_entity_embedding_list.append(zeros)
                    i_relation_embedding_list.append(zeros)
                    continue

                if i == self.path_len:
                    relation_embeddings = self.relation_embedding_matrix[[self.n_relation for path in paths]]
                else:
                    relation_embeddings = self.relation_embedding_matrix[[relation_dict[(path[i], path[i+1])] for path in paths]]

                entity_embeddings = self.entity_embedding_matrix[[path[i] for path in paths]]
                i_relation_embedding_list.append(relation_embeddings)
                i_entity_embedding_list.append(entity_embeddings)


            relations_embeddings = t.cat(i_relation_embedding_list, dim=0)
            entities_embeddings = t.cat(i_entity_embedding_list, dim=0)
            embeddings = t.cat([entities_embeddings, relations_embeddings], dim=-1).reshape(1, -1, 2 * self.dim)
            embeddings_list.append(embeddings)
        return embeddings_list

def get_hit(gt_item, pred_items):

    if gt_item in pred_items:
        return 1
    else:
        return 0


def get_ndcg(gt_item, pred_items):

    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    else:
        return 0

def eval_topk(model, rec, paths_dict, relation_dict, p, topk ):
    HR, NDCG = [], []
    model.eval()
    for user in rec:

        pairs = [(user, item, -1) for item in rec[user]]
        paths_list, _, users, items = get_data(pairs, paths_dict, p)

        predict_list = model(paths_list, relation_dict).cpu().detach().numpy().tolist()

        item_scores = {items[i]: predict_list[i] for i in range(len(pairs))}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: topk]
        HR.append(get_hit(items[-1], item_list))
        NDCG.append(get_ndcg(items[-1], item_list))

    model.train()
    return np.mean(HR), np.mean(NDCG)


def eval_topk1(model, rec, paths_dict, relation_dict, p):
    precision_list = []
    model.eval()
    for user in rec:
        pairs = [(user, item, -1) for item in rec[user]]
        paths_list, _, users, items = get_data(pairs, paths_dict, p)
        predict_list = model(paths_list, relation_dict).cpu().detach().numpy().tolist()
        item_scores = {items[i]: predict_list[i] for i in range(len(pairs))}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        precision_list.append([len({items[-1]}.intersection(item_list[: k])) / k for k in [1, 2, 3, 4, 5, 10, 20]])##.intersection返回多个集合（集合的数量大于等于2）的交集，
    model.train()
    return np.array(precision_list).mean(axis=0).tolist()


def eval_ctr(model, pairs, paths_dict, args, relation_dict):

    model.eval()
    pred_label = []
    paths_list, true_label, users, items = get_data(pairs, paths_dict, args.p)
    for i in range(0, len(pairs), args.batch_size):
        predicts = model(paths_list[i: i+args.batch_size], relation_dict)
        pred_label.extend(predicts.cpu().detach().numpy().tolist())
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return auc, acc


def get_data(pairs, paths_dict, p):
    paths_list = []
    label_list = []
    users = []
    items = []
    for pair in pairs:
        if len(paths_dict[(pair[0], pair[1])]):
            paths = paths_dict[(pair[0], pair[1])]

            if len(paths) >= p:
                indices = np.random.choice(len(paths), p, replace=False)
            else:
                indices = np.random.choice(len(paths), p, replace=True)

            paths_list.append([paths[i] for i in indices])
        else:
            paths_list.append([])

        label_list.append(pair[2])
        users.append(pair[0])
        items.append(pair[1])
    return paths_list, label_list, users, items


def train(args, is_topk):
    np.random.seed(123)
    data_dir = './data/' + args.dataset + '/'
    train_set = np.load(data_dir + str(args.ratio) + '_train_set.npy').tolist()
    eval_set = np.load(data_dir + str(args.ratio) + '_eval_set.npy').tolist()
    test_set = np.load(data_dir + str(args.ratio) + '_test_set.npy').tolist()
    entity_list = np.load(data_dir + '_entity_list.npy').tolist()
    relation_dict = np.load(data_dir + str(args.ratio) + '_relation_dict.npy', allow_pickle=True).item()
    _, _, n_relation = load_kg(data_dir)
    n_entity = len(entity_list)
    paths_dict = np.load(data_dir + str(args.ratio) + '_' + str(args.path_len) + '_path_dict.npy', allow_pickle=True).item()
    rec = np.load(data_dir + str(args.ratio) + '_rec.npy', allow_pickle=True).item()
    model = AGRE(args, n_entity, n_relation+2)
    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end=', ')
    print('p: %d' % args.p, end=', ')
    print('lr: %1.0e' % args.lr, end=', ')
    print('l2: %1.0e' % args.l2, end=', ')
    print('batch_size: %d' % args.batch_size)

    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    train_gauc_list = []
    test_gauc_list = []
    eval_gauc_list = []
    rec_gauc_list = []
    HR_list = []
    NDCG_list = []
    precision_list = []

    for epoch in range(args.epochs):
        loss_sum = 0
        np.random.shuffle(train_set)
        paths, true_label, users, items = get_data(train_set, paths_dict, args.p)
        labels = t.tensor(true_label).float()
        if t.cuda.is_available():
            labels = labels.to(args.device)
        start_index = 0
        size = len(paths)
        model.train()
        while start_index < size:
            predicts = model(paths[start_index: start_index + args.batch_size],
                             relation_dict)
            loss = criterion(predicts, labels[start_index: start_index + args.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()
            start_index += args.batch_size


        #train_auc, train_acc = eval_ctr(model, train_set, paths_dict, args, relation_dict)
        #eval_auc, eval_acc = eval_ctr(model, eval_set, paths_dict, args, relation_dict)
        test_auc, test_acc = eval_ctr(model, test_set, paths_dict, args, relation_dict)
        #train_gauc = cal_group_auc(model, train_set, paths_dict, args, relation_dict)
        #eval_gauc = cal_group_auc(model, eval_set, paths_dict, args, relation_dict)
        test_gauc = cal_group_auc(model, test_set, paths_dict, args, relation_dict)

        rec_gauc = group_auc(model, rec, paths_dict, relation_dict, args)
        print((epoch + 1), '\t', test_gauc, '\t', rec_gauc, '\t', test_auc, '\t', test_acc, end='\t')
        HR, NDCG = 0, 0
        precisions = []
        if is_topk:
            HR, NDCG = eval_topk(model, rec, paths_dict, relation_dict, args.p, args.topk)
            print(HR, '\t',NDCG, end='\t')
            precisions = eval_topk1(model, rec, paths_dict, relation_dict, args.p)
            print("Precision: ", end='[')

            for i in range(len(precisions)):

                if i == len(precisions) - 1:
                    print("%.4f" % precisions[i], end=']'+'\n')
                else:
                    print("%.4f" % precisions[i], end=', ')
        # train_auc_list.append(train_auc)
        # train_acc_list.append(train_acc)
        # eval_auc_list.append(eval_auc)
        # eval_acc_list.append(eval_acc)
        # train_gauc_list.append(train_gauc)
        # eval_gauc_list.append(eval_gauc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        test_gauc_list.append(test_gauc )
        rec_gauc_list.append(rec_gauc)
        HR_list.append(HR)
        NDCG_list.append(NDCG)
        precision_list.append(precisions)

    return 0

def cal_group_auc(model, pairs, paths_dict, args, relation_dict):
    """Calculate group auc"""
    model.eval()
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    group_score1 = defaultdict(lambda: [])
    group_truth1 = defaultdict(lambda: [])
    score_truth = defaultdict(lambda: [])
    pred_label = []
    paths_list, true_label, users, items = get_data(pairs, paths_dict, args.p)
    for i in range(0, len(pairs), args.batch_size):
        predicts = model(paths_list[i: i + args.batch_size], relation_dict)
        pred_label.extend(predicts.cpu().detach().numpy().tolist())
    for i in range(len(users)):
        user_id = users[i]
        score = pred_label[i]
        truth = true_label[i]
        group_score1[user_id].append(score)
        group_truth1[user_id].append(truth)
        key_list = list(group_score1.keys())
    for i in range(len(key_list)):
        user_id = key_list[i]
        for j in range(len(group_score1[user_id])):
            score_truth[group_score1[user_id][j]] = group_truth1[user_id][j]
        score_truth = dict([(m, n) for m, n in sorted(score_truth.items())])
        score_list = list(score_truth.keys())
        truth_list = list(score_truth.values())
        for k in range(len(score_list)):
            group_score[user_id].append(score_list[k])
            group_truth[user_id].append(truth_list[k])
        score_truth.clear()
    group_flag = defaultdict(lambda: False)

    for user_id in set(users):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc

def group_auc(model, rec, paths_dict, relation_dict, args):
    model.eval()
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for user in rec:
        pairs = [(user, item, 0) for item in rec[user]]
        paths_list, _, users, items = get_data(pairs, paths_dict, args.p)
        items1 = set(items)
        items = list(items1)
        predict_list = model(paths_list, relation_dict).cpu().detach().numpy().tolist()
        item_scores = {items[i]: predict_list[i] for i in range(len(items))}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: args.topk]
        if items[-1] in item_list:
            _[item_list.index(items[-1])] = 1

        for item in item_list:
            group_score[user].append(item_scores[item])
        group_truth[user] = _[:args.topk]

    group_flag = defaultdict(lambda: False)

    for user_id in rec:
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0

    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc


