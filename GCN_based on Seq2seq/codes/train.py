# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch

import numpy as np
import pickle
import scipy.sparse as sp
from collections import deque


class RnnParameterData(object):
    def __init__(self,
                 loc_graph_emb_size=400,
                 loc_emb_size=300,
                 hidden_size=300,
                 lr=1e-4,
                 lr_step=3,
                 lr_decay=0.5,
                 dropout_p=0.6,
                 L2=1e-7,
                 clip=3.0,
                 optim='Adam',
                 rnn_type='LSTM',
                 data_path='./data/',
                 save_path='./results/',
                 data_name='foursquare_2012'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(
            open(self.data_path + self.data_name + '.pk', 'rb'),
            encoding='iso-8859-1')
        self.vid_look_up = data['vid_lookup']
        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']

        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.loc_graph_emb_size = loc_graph_emb_size
        self.loc_emb_size = loc_emb_size
        self.hidden_size = hidden_size

        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip
        self.rnn_type = rnn_type


def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    adj_train = {}
    train_idx = {}
    loc_train = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        adj_train[u] = {}
        loc_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            if len(target) == 1:
                pass

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])

            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_tmp = np.array([s[0] for s in loc_tim])
            loc_tmp = loc_tmp[:-len(target) + 1]
            loc_np = np.reshape(
                np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(
                np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            loc_map = {}
            for _, z in enumerate(loc_tmp):
                if z not in loc_map:
                    loc_map[z] = len(loc_map)
            loc_array = np.array(list(loc_map.keys()))

            #################################################
            edges_list = []
            for m, _ in enumerate(loc_tmp):
                if m == 0:
                    continue
                edges_list.append([loc_tmp[m - 1], loc_tmp[m]])
            edges = np.array(
                list(map(loc_map.get,
                         np.array(edges_list).flatten())),
                dtype=np.int32).reshape(np.array(edges_list).shape)
            adj = sp.coo_matrix(
                (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                shape=(len(loc_map), len(loc_map)),
                dtype=np.float32)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            trace['loc'] = torch.LongTensor(loc_np)
            trace['tim'] = torch.LongTensor(tim_np)
            trace['target'] = torch.LongTensor(target)
            loc_array = torch.LongTensor(loc_array)
            data_train[u][i] = trace
            adj_train[u][i] = adj
            loc_train[u][i] = loc_array
        train_idx[u] = train_id
    return data_train, train_idx, adj_train, loc_train


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def generate_adj_input(loc_arr, loc_map, adj_input):
    loc_num = len(loc_arr)
    adj = np.zeros((loc_num, loc_num))
    adj[:] = np.inf
    for loc_1 in loc_arr:
        for loc_2 in loc_arr:
            try:
                if loc_2 >= loc_1:
                    adj[loc_map[loc_1]][loc_map[loc_2]] = adj_input[loc_1 - 1][
                        loc_2 - 1]
            except IndexError:
                print(loc_1, loc_2)

    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = train_idx.keys()
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        list_user = list(user)
        while queue_left > 0:
            np.random.shuffle(list_user)
            for j, u in enumerate(list_user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum(
                [1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue