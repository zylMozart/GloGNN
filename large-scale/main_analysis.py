from dataset import load_nc_dataset
from homophily import our_measure, edge_homophily_edge_idx, each_node_homophily, node_homophily, edge_homophily, attribute_homophily
from sklearn.cluster import KMeans
import torch
import numpy as np
import pandas as pd
import os
os.chdir('large-scale')
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib.pyplot as plt

# datasets = '''fb100 chameleon cornell film'''.split(' ')
datasets = '''Cora CiteSeer PubMed chameleon cornell film squirrel texas wisconsin fb100 pokec arxiv-year snap-patents genius twitch-gamer yelp-chi'''.split(' ')

def graph_hom_measurement():
    res = {'node_hom':[],'edge_hom':[],'node_hom_kmeans':[],'edge_hom_kmeans':[],'attr_hom':[],}
    for dataset_name in datasets:
        dataset = load_nc_dataset(dataset_name)
        # Edge&Node Hom
        hom_edge = edge_homophily(dataset.graph['edge_index'],dataset.label)
        hom_node = each_node_homophily(None, dataset.label, dataset.graph['edge_index'], None)
        hom_node = float(hom_node[torch.logical_not(hom_node.isnan())].mean())
        hom_attr = attribute_homophily(dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.label.shape[0])
        res['node_hom'].append(hom_node)
        res['edge_hom'].append(hom_edge)
        print('{:<12} Edge_Hom: {:.4f} Node_Hom: {:.4f} Attr_Hom: {:.4f}'.format(dataset_name, hom_edge, hom_node, hom_attr))
        # K-means
        kmeans_model = KMeans(n_clusters=dataset.label.unique().shape[0], random_state=1).fit(dataset.graph['node_feat'])
        labels = torch.Tensor(kmeans_model.labels_)
        # Edge&Node Hom
        hom_edge = edge_homophily(dataset.graph['edge_index'],labels)
        hom_node = each_node_homophily(None, labels, dataset.graph['edge_index'], None)
        hom_node = float(hom_node[torch.logical_not(hom_node.isnan())].mean())
        res['node_hom_kmeans'].append(hom_node)
        res['edge_hom_kmeans'].append(hom_edge)
        print('{:<12} Edge_Hom: {:.4f} Node_Hom: {:.4f}'.format(dataset_name+'*', hom_edge, hom_node))
    res['attr_hom'] = [(i-min(res['attr_hom'])/(max(res['attr_hom'])-min(res['attr_hom']))) for i in res['attr_hom']]
    pd_res = pd.DataFrame(list(zip(datasets,res['node_hom'],res['edge_hom'],res['node_hom_kmeans'],res['edge_hom_kmeans'],res['attr_hom'])),columns=[['dataset']+list(res.keys())])
    pd_res.to_csv('results/hom/statics.csv')
    # Sort with Node hom
    sort_index = np.argsort(res['node_hom'])
    for k,v in res.items():
        res[k] = [v[i] for i in sort_index]
    num = len(datasets)
    x = list(range(num))
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, res['node_hom'], label="Node Hom")
    ax.plot(x, res['node_hom_kmeans'], label="Kmeans Node Hom")
    ax.plot(x, res['edge_hom_kmeans'], label="Kmeans Edge Hom")
    ax.legend()
    plt.savefig('results/hom/hom_measurement.png')
    pass


if __name__=='__main__':
    graph_hom_measurement()