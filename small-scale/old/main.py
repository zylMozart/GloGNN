from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter
import math
import numpy as np
import scipy.sparse as sp
import time

import sys
import pickle as pkl
import networkx as nx
from collections import defaultdict
import os
import json
import warnings
from datetime import datetime
import pandas as pd
from model import MLP_NORM, GCN
from config import get_parser

if 'small-scale' not in os.getcwd(): os.chdir('small-scale')
warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)


cpu_num = int(cpu_count() * 0.95)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)




def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Read split data
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data_new(dataset_str, split):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # print('dataset_str', dataset_str)
    # print('split', split)
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = list(np.where(train_mask == 1)[0])
        idx_val = list(np.where(val_mask == 1)[0])
        idx_test = list(np.where(test_mask == 1)[0])

        no_label_nodes = []
        if dataset_str == 'citeseer':  # citeseer has some data with no label
            for i in range(len(labels)):
                if sum(labels[i]) < 1:
                    labels[i][0] = 1
                    no_label_nodes.append(i)

            for n in no_label_nodes:  # remove unlabel nodes from train/val/test
                if n in idx_train:
                    idx_train.remove(n)
                if n in idx_val:
                    idx_val.remove(n)
                if n in idx_test:
                    idx_test.remove(n)

    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_str,
                                                                f'out1_node_feature_label.txt')
        graph_dict = defaultdict(list)
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                graph_dict[int(line[0])].append(int(line[1]))
                graph_dict[int(line[1])].append(int(line[0]))

        # print(sorted(graph_dict))
        graph_dict_ordered = defaultdict(list)
        for key in sorted(graph_dict):
            graph_dict_ordered[key] = graph_dict[key]
            graph_dict_ordered[key].sort()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
        # adj = sp.csr_matrix(adj)

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]

    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # print('adj', adj.shape)
    # print('features', features.shape)
    # print('labels', labels.shape)
    # print('idx_train', idx_train.shape)
    # print('idx_val', idx_val.shape)
    # print('idx_test', idx_test.shape)
    return adj, features, labels, idx_train, idx_val, idx_test


# Training settings
parser = get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data_new(
    args.dataset, args.split)


# Change data type to float
features = features.to(torch.float64)
adj = adj.to(torch.float64)

# Model and optimizer

if args.model == 'gcn':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
elif args.model == 'mlp_norm':
    model = MLP_NORM(
        nnodes=adj.shape[0],
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=labels.max().item() + 1,
        dropout=args.dropout,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        norm_func_id=args.norm_func_id,
        norm_layers=args.norm_layers,
        orders=args.orders,
        orders_func_id=args.orders_func_id,
        cuda=args.cuda)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# Train model
cost_val = []
t_total = time.time()

num_nodes = labels.shape[0]
num_labels = labels.unique().shape[0]

# print(model.diag_weight)
start = True
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    
    if start==False and epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping//2+1):-1]):
        cost_val = [np.inf for i in cost_val]
        start = True
    if start:
        print('*',end='')
        HET_THRESHOLD = 0.8
        TEMP = 1.0
        LAMBDA = 0.1
        # 1. Define hetero graph
        hetero = torch.zeros((num_nodes,num_labels))
        preds = output.max(1)[1]+1 # Avoid 0 in the next matrix mul
        adj_pred = adj.clone().to_dense()
        adj_pred[adj_pred>0]=1
        adj_pred[adj_pred==0]=-1
        adj_pred = adj_pred * preds.repeat(num_nodes, 1)
        for i_label in range(num_labels):
            hetero[:,i_label] = (adj_pred==(i_label+1)).sum(dim=1)
        hetero = hetero / hetero.norm(dim=1)[:, None]
        similarity_matrix = torch.mm(hetero, hetero.t())
        CL_adj = (similarity_matrix>HET_THRESHOLD).int().to('cuda')
        CL_adj = CL_adj.fill_diagonal_(1)
        CL_adj.requires_grad = False
        # 2. CL loss
        z = F.normalize(output)
        s = torch.mm(z,z.t())
        sim = torch.exp(s/TEMP).to('cuda')
        loss_CL = LAMBDA*sim*CL_adj
        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + loss_CL.mean()
        # loss_train = loss_CL.mean()
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # print(model.diag_weight)

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    cost_val.append(loss_val.item())
    if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
        # print("Early stopping...")
        break
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

outfile_name = f"{args.dataset}_lr{args.lr}_do{args.dropout}_es{args.early_stopping}_" +\
    f"wd{args.weight_decay}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_" +\
    f"delta{args.delta}_nlid{args.norm_func_id}_nl{args.norm_layers}_" +\
    f"ordersid{args.orders_func_id}_orders{args.orders}_split{args.split}_results.txt"
print(outfile_name)

# Testing
model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))

test_time = time.time()
results_dict = {}
results_dict['test_cost'] = float(loss_test.item())
results_dict['test_acc'] = float(acc_test.item())
results_dict['test_duration'] = time.time()-test_time


# outfile_name = f'''{args.dataset}_split{args.split}_results.txt'''

with open(os.path.join('runs', outfile_name), 'w') as outfile:
    outfile.write(json.dumps(results_dict))

# out = model(features, adj)
# out = out.argmax(dim=-1, keepdim=True).detach().cpu()
# torch.save(out, f'/home/yilun/HOM_GNN/GloGNN/large-scale/results/hom/pred_{args.model}_{args.dataset}.pt')

# result=results_dict
# result.update(vars(args))
# result['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
# df = pd.DataFrame(columns=result.keys())
# df = df.append(result, ignore_index=True)
# save_path = '../res/hetero/compare_hetero.csv'
# if os.path.exists(save_path):
#     df.to_csv(save_path,mode='a',header=False) 
# else:
#     df.to_csv(save_path,mode='w',header=True) 