
import torch
import numpy as np
import scipy.sparse as sp
import tqdm
import networkx
import time
import copy
import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import normalize as sk_normalize
from struct_sim import graph, struc2vec
from torch_geometric.utils import to_networkx
#####################################################################################
# used in GGCN


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)


def precompute_degree_s(adj):
    adj_i = adj._indices()
    adj_v = adj._values()
    # print('adj_i', adj_i.shape)
    # print(adj_i)
    # print('adj_v', adj_v.shape)
    # print(adj_v)
    adj_diag_ind = (adj_i[0, :] == adj_i[1, :])
    adj_diag = adj_v[adj_diag_ind]
    # print(adj_diag)
    # print(adj_diag[0])
    v_new = torch.zeros_like(adj_v)
    for i in tqdm(range(adj_i.shape[1])):
        # print('adj_i[0,', i, ']', adj_i[0, i])
        v_new[i] = adj_diag[adj_i[0, i]]/adj_v[i]-1
    degree_precompute = torch.sparse.FloatTensor(
        adj_i, v_new, adj.size())
    return degree_precompute


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high

#####################################################################################
# used in wrgat


def build_struc_layers(G, opt1=True, opt2=True, opt3=True, until_layer=None, workers=64):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if(opt3):
        until_layer = until_layer
    else:
        until_layer = None

    G = struc2vec.Graph(G, False, workers, untilLayer=until_layer)

    if(opt1):
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if(opt2):
        G.create_vectors()
        G.calc_distances(compactDegree=opt1)
    else:
        G.calc_distances_all_vertices(compactDegree=opt1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()
    return


def build_multigraph_from_layers(networkx_graph, y, x=None):
    num_of_nodes = networkx_graph.number_of_nodes()

    x_degree = torch.zeros(num_of_nodes, 1)
    for i in range(0, num_of_nodes):
        x_degree[i] = torch.Tensor([networkx_graph.degree(i)])

    inp = open("struc_sim/pickles/distances_nets_graphs.pickle", "rb")
    distances_nets_graphs = pickle.load(inp, encoding="bytes")
    src = []
    dst = []
    edge_weight = []
    edge_color = []
    for layer, layergraph in distances_nets_graphs.items():
        filename = "struc_sim/pickles/distances_nets_weights-layer-" + \
            str(layer) + ".pickle"
        inp = open(filename, "rb")
        distance_nets_weights_layergraph = pickle.load(inp, encoding="bytes")
        for node_id, nbd_ids in layergraph.items():
            s = list(np.repeat(node_id, len(nbd_ids)))
            d = nbd_ids
            src += s
            dst += d
            edge_weight += distance_nets_weights_layergraph[node_id]
            edge_color += list(np.repeat(layer, len(nbd_ids)))
        assert len(src) == len(dst) == len(edge_weight) == len(edge_color)

    edge_index = np.stack((np.array(src), np.array(dst)))
    edge_weight = np.array(edge_weight)
    edge_color = np.array(edge_color)

    # print(edge_index.shape)
    # print(edge_weight.shape)
    if x is None:
        data = Data(x=x_degree, edge_index=torch.LongTensor(edge_index), edge_weight=torch.FloatTensor(edge_weight),
                    edge_color=torch.LongTensor(edge_color), y=y)
    else:
        data = Data(x=x, x_degree=x_degree, edge_index=torch.LongTensor(edge_index),
                    edge_weight=torch.FloatTensor(edge_weight),
                    edge_color=torch.LongTensor(edge_color), y=y)

    return data


def build_pyg_struc_multigraph(pyg_data):
    # print("Start build_pyg_struc_multigraph")
    start_time = time.time()
    # print(pyg_data)
    G = graph.from_pyg(pyg_data)
    # print('Before G', G)
    networkx_graph = to_networkx(pyg_data)
    # print('networkx_graph', networkx_graph)
    print("Done converting to networkx")
    build_struc_layers(G)
    # print('After G', G)
    print("Done building layers")
    data = build_multigraph_from_layers(networkx_graph, pyg_data.y, pyg_data.x)
    # print(data)
    if hasattr(pyg_data, 'train_mask'):
        data.train_mask = pyg_data.train_mask
        data.val_mask = pyg_data.val_mask
        data.test_mask = pyg_data.test_mask
    time_cost = time.time() - start_time
    print("build_pyg_struc_multigraph cost: ", time_cost)
    return data


def filter_rels(data, r):
    data = copy.deepcopy(data)
    mask = data.edge_color <= r
    data.edge_index = data.edge_index[:, mask]
    data.edge_weight = data.edge_weight[mask]
    data.edge_color = data.edge_color[mask]
    return data


def structure_edge_weight_threshold(data, threshold):
    data = copy.deepcopy(data)
    mask = data.edge_weight > threshold
    data.edge_weight = data.edge_weight[mask]
    data.edge_index = data.edge_index[:, mask]
    data.edge_color = data.edge_color[mask]
    return data


def add_original_graph(og_data, st_data, weight=1.0):
    st_data = copy.deepcopy(st_data)
    e_i = torch.cat((og_data.edge_index, st_data.edge_index), dim=1)
    st_data.edge_color = st_data.edge_color + 1
    e_c = torch.cat((torch.zeros(
        og_data.edge_index.shape[1], dtype=torch.long), st_data.edge_color), dim=0)
    e_w = torch.cat((torch.ones(
        og_data.edge_index.shape[1], dtype=torch.float)*weight, st_data.edge_weight), dim=0)
    st_data.edge_index = e_i
    st_data.edge_color = e_c
    st_data.edge_weight = e_w
    return st_data

#####################################################################################
