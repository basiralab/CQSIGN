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
import torch
import os

def save_data(x, y, edge_index, train_mask, val_mask, test_mask, directory):
    np.save(directory + '/data_feat.npy',x)
    np.save(directory + '/data_label.npy',y)
    np.save(directory + '/edge_index.npy',edge_index)
    np.save(directory + '/train_mask.npy',train_mask)
    np.save(directory + '/val_mask.npy',val_mask)
    np.save(directory + '/test_mask.npy',test_mask)
    
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', nargs='?', const='PPI', type=str, default='PPI')
parser.add_argument('--centrality', nargs='?', const='EC', type=str, default='EC')
parser.add_argument('--node_budget', nargs='?', const=15000, type=int, default=15000)

print("Generating GNN AutoScale Dataset for:")
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
    
# split graph to train, val, and test (inductive training)
(x_train, y_train, edge_train, train_mask, x_val, y_val, edge_val, val_mask, x_test, 
 y_test, edge_test, test_mask) = utility.split_graph(G, multilabel = multilabel)

path = "OtherBenchmarks/GNNAutoScale/pyg_autoscale/large_benchmark/data/"
suffix = dataset_name + '_' + centrality

# save training data
edge_train = torch.tensor(edge_train)
edge_train = torch.transpose(edge_train,0,1)
edge_train_flip = torch.flip(edge_train,[0]) # re-adds flipped edges that were removed by networkx
edge_train = torch.cat((edge_train, edge_train_flip), 1).numpy()

train_mask_train = train_mask
val_mask_train = np.full((x_train.shape[0],),False)
test_mask_train = np.full((x_train.shape[0],),False)

directory = path + suffix + '/train/raw'
if not os.path.exists(directory):
   os.makedirs(directory)
   
save_data(x_train, y_train, edge_train, train_mask_train, val_mask_train, 
          test_mask_train, directory)

# save validation data
edge_val = torch.tensor(edge_val)
edge_val = torch.transpose(edge_val,0,1)
edge_val_flip = torch.flip(edge_val,[0]) # re-adds flipped edges that were removed by networkx
edge_val = torch.cat((edge_val, edge_val_flip), 1).numpy()

train_mask_val = np.full((x_val.shape[0],),False)
val_mask_val = val_mask
test_mask_val = np.full((x_val.shape[0],),False)

directory = path + suffix + '/val/raw'
if not os.path.exists(directory):
   os.makedirs(directory)
   
save_data(x_val, y_val, edge_val, train_mask_val, val_mask_val, 
          test_mask_val, directory)

# save test data
edge_test = torch.tensor(edge_test)
edge_test = torch.transpose(edge_test,0,1)
edge_test_flip = torch.flip(edge_test,[0]) # re-adds flipped edges that were removed by networkx
edge_test = torch.cat((edge_test, edge_test_flip), 1).numpy()

train_mask_test = np.full((x_test.shape[0],),False)
val_mask_test = np.full((x_test.shape[0],),False)
test_mask_test = test_mask

directory = path + suffix + '/test/raw'
if not os.path.exists(directory):
   os.makedirs(directory)
   
save_data(x_test, y_test, edge_test, train_mask_test, val_mask_test, 
          test_mask_test, directory)
