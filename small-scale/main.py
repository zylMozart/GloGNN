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
from load_data import load_data_new
from utils import accuracy, seed_everything, set_cpu_num

if 'small-scale' not in os.getcwd(): os.chdir('small-scale')
warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

# settings
set_cpu_num()
parser = get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
seed_everything(args.seed)


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
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
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

# with open(os.path.join('runs', outfile_name), 'w') as outfile:
#     outfile.write(json.dumps(results_dict))

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