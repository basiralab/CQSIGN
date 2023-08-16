#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:55:32 2023

@author: chris
"""

import numpy as np
from networkx.readwrite import json_graph
import contraction
import utility
import datasets
import networkx as nx
import json
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', nargs='?', const='PPI', type=str, default='PPI')
parser.add_argument('--centrality', nargs='?', const='EC', type=str, default='EC')
parser.add_argument('--node_budget', nargs='?', const=15000, type=int, default=15000)

print("Generating Cluster-GCN Dataset for:")
args = parser.parse_args()
print(args)

dataset_name = args.dataset_name
centrality = args.centrality
steps = [args.node_budget]

if dataset_name == 'PPI':
    multilabel = True
    
    # get train dataset
    (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_PPI(split='train')
    num_features = x.shape[1]
    num_classes = y.shape[1]

    # get initial label distribution
    Y_dist_before = utility.get_label_distribution_tensor(y[train_mask], multilabel)

    # construct train graph
    G_train = utility.construct_graph(x, y, edge_index, train_mask, val_mask, test_mask)

    # get val dataset
    (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_PPI(split='val')

    # construct val graph
    G_val = utility.construct_graph(x, y, edge_index, train_mask, val_mask, test_mask)

    # get test dataset
    (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_PPI(split='test')

    # construct test graph
    G_test = utility.construct_graph(x, y, edge_index, train_mask, val_mask, test_mask)

    # contract training graph
    G_train = contraction.contract_graph(G_train, centrality = centrality, 
                                           num_features = num_features, 
                                           num_classes = num_classes,
                                           steps = steps, multilabel = multilabel)
    
    # combine train, val, and test to single graph
    G_val = nx.convert_node_labels_to_integers(G_val, first_label=G_train.number_of_nodes(), ordering='default')
    G_test = nx.convert_node_labels_to_integers(G_test, first_label=G_train.number_of_nodes() + G_val.number_of_nodes(), ordering='default')
    
    G = nx.compose(G_train, G_val)
    G = nx.compose(G, G_test)

else:
    multilabel = False
    
    if dataset_name == 'OrganC':
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_Organ(view='C')
        
    elif dataset_name == 'OrganS':
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_Organ(view='S')
    
    elif dataset_name == 'Flickr':
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_Flickr()
        
    num_features = x.shape[1]
    num_classes = max(y) + 1
    
    # construct networkx graph
    G = utility.construct_graph(x, y, edge_index, train_mask, val_mask, 
                                    test_mask)
    
    # contract graph
    G = contraction.contract_graph(G, centrality = centrality, 
                                   num_features = num_features, 
                                   num_classes = num_classes,
                                   steps = steps, multilabel = multilabel)

# Reordering the node key to train -> val -> test
reorder_map = {}
curr_id = 0
for node in G.nodes(data=True):
    if bool(node[1]['train']) == True:
        reorder_map[node[0]] = curr_id
        curr_id += 1
for node in G.nodes(data=True):
    if bool(node[1]['val']) == True:
        reorder_map[node[0]] = curr_id
        curr_id += 1
for node in G.nodes(data=True):
    if bool(node[1]['test']) == True:
        reorder_map[node[0]] = curr_id
        curr_id += 1
G = nx.relabel_nodes(G, reorder_map)

# Generate Cluster-GCN Data Format
x = np.empty((G.number_of_nodes(), num_features))
y = {}
new_id_map = {}
new_G = nx.Graph()

for node in G.nodes(data=True):
    x[node[0],:] = node[1]['x']
    y[f"{node[0]}"] = node[1]['y'].astype(int).tolist()
    new_id_map[f"{node[0]}"] = node[0]
    new_G.add_node(node[0])
    new_G.nodes[node[0]]['val'] = bool(node[1]['val'])
    new_G.nodes[node[0]]['test'] = bool(node[1]['test'])

new_G.add_edges_from(G.edges())
    
path = "OtherBenchmarks/Cluster-GCN/cluster_gcn/data/"
suffix = dataset_name + '_' + centrality

# create directory
if not os.path.exists(path + suffix):
   os.makedirs(path + suffix)
   
# saving class map JSON
file = open(path + suffix + '/' + suffix + '-class_map.json','w')
json.dump(y,file)
file.close()

# saving id map JSON
file = open(path + suffix + '/' + suffix + '-id_map.json','w')
json.dump(new_id_map,file)
file.close()

# saving JSON graph file
file = open(path + suffix + '/' + suffix + '-G.json', 'w')
json.dump(json_graph.node_link_data(new_G),file)
file.close()

# saving fature npy file
np.save(path + suffix + '/' + suffix + '-feats.npy',x)
