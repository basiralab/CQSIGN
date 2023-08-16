#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:57:27 2023

@author: chris
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import networkx as nx
import torch
from torch_geometric.typing import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def construct_graph(x, y, edge_index, train_mask, val_mask, test_mask):
    
    # Construct NetworkX Graph
    nodes = [i for i in range(x.shape[0])]
    G = nx.Graph()
    for i in nodes:
        G.add_node(i, x = x[i], y = y[i], train = train_mask[i], 
                   val = val_mask[i], test = test_mask[i])
    G.add_edges_from(edge_index)
    
    return G

def construct_normalized_adj(edge_index, num_nodes):
    
    edge_index = torch.tensor(edge_index)
    edge_index = torch.transpose(edge_index,0,1)
    edge_index_flip = torch.flip(edge_index,[0]) # re-adds flipped edges that were removed by networkx
    edge_index = torch.cat((edge_index, edge_index_flip), 1)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes,num_nodes))
    adj = adj.set_diag() # adding self loops
    adj = gcn_norm(adj, add_self_loops=False) # normalization

    return adj

def convert_graph_to_tensor(G, multilabel = True):
    x = np.empty((G.number_of_nodes(),G.nodes[0]['x'].shape[0]))
    if multilabel:
        y = np.empty((G.number_of_nodes(),G.nodes[0]['y'].shape[0]),dtype = 'int')
    else:
        y = np.empty((G.number_of_nodes(),),dtype = 'int')
        
    edge_index = np.array([edge for edge in G.edges()])
    train_mask = np.empty((G.number_of_nodes(),),dtype = 'bool')
    val_mask = np.empty((G.number_of_nodes(),),dtype = 'bool')
    test_mask = np.empty((G.number_of_nodes(),),dtype = 'bool')
    
    for node in G.nodes(data=True):
        x[node[0],:] = node[1]['x']
        if multilabel:
            y[node[0],:] = node[1]['y']
        else:
            y[node[0]] = node[1]['y']
        
        train_mask[node[0]] = node[1]['train']
        val_mask[node[0]] = node[1]['val']
        test_mask[node[0]] = node[1]['test']
    
    return x, y, edge_index, train_mask, val_mask, test_mask

def split_graph(G, multilabel = True):
    print("Splitting Graph...")
    print("=============== Graph Splitting ===============")
    
    # Get complete test graph
    x_test, y_test, edge_test, _, _, test_mask = convert_graph_to_tensor(G, multilabel=multilabel)
    
    print(f"Unlabeled + Test + Validation + Training graph nodes: {x_test.shape[0]}")
    print(f"Unlabeled + Test + Validation + Training graph edges: {edge_test.shape[0]}")
    print(f"Total test nodes: {test_mask.sum()}")
    
    # Get training + val graph
    # remove all test nodes
    test_nodes = []
    for node in G.nodes(data=True):
        if node[1]['test']:
            test_nodes.append(node[0])
    G.remove_nodes_from(test_nodes)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    x_val, y_val, edge_val, _, val_mask, _ = convert_graph_to_tensor(G, multilabel=multilabel)
    
    print(f"Unlabeled + Validation + Training graph nodes: {x_val.shape[0]}")
    print(f"Unlabeled + Validation + Training graph edges: {edge_val.shape[0]}")
    print(f"Total val nodes: {val_mask.sum()}")
    
    # Get training graph
    # remove all val nodes
    val_nodes = []
    for node in G.nodes(data=True):
        if node[1]['val']:
            val_nodes.append(node[0])
    G.remove_nodes_from(val_nodes)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    
    x_train, y_train, edge_train, train_mask, _, _ = convert_graph_to_tensor(G, multilabel = multilabel)
    
    print(f"Unlabeled + Training graph nodes: {x_train.shape[0]}")
    print(f"Unlabeled + Training graph edges: {edge_train.shape[0]}")
    print(f"Total train nodes: {train_mask.sum()}")
    print()
    
    return (x_train, y_train, edge_train, train_mask, x_val, y_val, edge_val, 
            val_mask, x_test, y_test, edge_test, test_mask)

def get_label_distribution(G, multilabel = True):
    
    Y = []
    for node in G.nodes(data=True):
        Y.append(node[1]['y'])
        
    if multilabel:
        num_classes = Y[0].shape[0] # logits for multi label
        Y_dist = [0 for i in range(num_classes)]
        for i in range(len(Y)):
            for j in range(num_classes):
                Y_dist[j] += Y[i][j]/len(Y)
        
    else:
        num_classes = max(Y) + 1 # class for multi class
        Y_dist = [0 for i in range(num_classes)]
        for i in range(len(Y)):
            Y_dist[Y[i]] += 1/len(Y)
            
    return Y_dist

def get_label_distribution_tensor(y, multilabel = True):
        
    Y = []
    for i in range(y.shape[0]):
        Y.append(y[i])
        
    if multilabel:
        num_classes = Y[0].shape[0] # logits for multi label
        Y_dist = [0 for i in range(num_classes)]
        for i in range(len(Y)):
            for j in range(num_classes):
                Y_dist[j] += Y[i][j]/len(Y)
        
    else:
        num_classes = max(Y) + 1 # class for multi class
        Y_dist = [0 for i in range(num_classes)]
        for i in range(len(Y)):
            Y_dist[Y[i]] += 1/len(Y)
            
    return Y_dist

def compute_label_distribution_error(Y_dist_before, Y_dist_after):
    
    Y_dist_error = []
    for i in range(len(Y_dist_before)):
        Y_dist_error.append(abs(Y_dist_after[i] - Y_dist_before[i])/Y_dist_before[i])
        
    return Y_dist_error

def logit_to_label(out):
    return out.argmax(dim=1)

def metrics(logits, y):
    
    if y.dim() == 1: # Multi-class
        y_pred = logit_to_label(logits)
        cm = confusion_matrix(y.cpu(),y_pred.cpu())
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
    
        acc = np.diag(cm).sum() / cm.sum()
        micro_f1 = acc # micro f1 = accuracy for multi-class
        sens = TP.sum() / (TP.sum() + FN.sum())
        spec = TN.sum() / (TN.sum() + FP.sum())
    
    else: # Multi-label
        y_pred = logits >= 0
        y_true = y >= 0.5
        
        tp = int((y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        
        acc = (tp + tn)/(tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        sens = (tp)/(tp + fn)
        spec = (tn)/(tn + fp)
        
    return acc, micro_f1, sens, spec