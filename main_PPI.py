#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:29:06 2023

@author: chris
"""

import datasets
import utility
import contraction
import torch
import numpy as np
import models
import train_SIGN
import train_GCN
import writer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', nargs='?', const='QSIGN', type=str, default='QSIGN')
parser.add_argument('--centrality', nargs='?', const='EC', type=str, default='EC')
parser.add_argument('--node_budget', nargs='?', const=15000, type=int, default=15000)
parser.add_argument('--max_hop', nargs='?', const=2, type=int, default=2)
parser.add_argument('--layers', nargs='?', const=3, type=int, default=3)
parser.add_argument('--hidden_channels', nargs='?', const=1024, type=int, default=1024)
parser.add_argument('--dropout', nargs='?', const=0.2, type=float, default=0.2)
parser.add_argument('--batch_norm', nargs='?', const=True, type=bool, default=True)
parser.add_argument('--lr', nargs='?', const=0.005, type=float, default=0.005)
parser.add_argument('--num_batch', nargs='?', const=10, type=int, default=10)
parser.add_argument('--num_epoch', nargs='?', const=1000, type=int, default=1000)
parser.add_argument('--multilabel', nargs='?', const=True, type=bool, default=True)
parser.add_argument('--do_eval', nargs='?', const=True, type=bool, default=True)
parser.add_argument('--residual', nargs='?', const=True, type=bool, default=True)
parser.add_argument('--print_result', nargs='?', const=True, type=bool, default=True)

args = parser.parse_args()
print(args)

model_type = args.model_type
centrality = args.centrality
steps = [args.node_budget]
max_hop = args.max_hop
layers = args.layers
hidden_channels = args.hidden_channels
dropout = args.dropout
batch_norm = args.batch_norm
lr = args.lr
num_batch = args.num_batch
num_epoch = args.num_epoch
multilabel = args.multilabel
do_evaluation = args.do_eval
residual = args.residual
print_result = args.print_result

dataset_name = 'PPI'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Prepare Dataset
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

#%% Contract Graph
# contract training graph
G_train = contraction.contract_graph(G_train, centrality = centrality, 
                                       num_features = num_features, 
                                       num_classes = num_classes,
                                       steps = steps, multilabel = multilabel)

#%% Convert Graph to Tensor
# convert train graph to tensor
x_train, y_train, edge_train, _, _, _ = utility.convert_graph_to_tensor(G_train, multilabel=multilabel)

# convert val graph to tensor
x_val, y_val, edge_val, _, _, _ = utility.convert_graph_to_tensor(G_val, multilabel=multilabel)

# convert test graph to tensor
x_test, y_test, edge_test, _, _, _ = utility.convert_graph_to_tensor(G_test, multilabel=multilabel)

# get post-contraction label distribution
Y_dist_after = utility.get_label_distribution_tensor(y_train, multilabel)

# get label distribution errors
Y_dist_error = utility.compute_label_distribution_error(Y_dist_before, Y_dist_after)
print(f"Contraction Label Distribution Avg Error: {np.mean(Y_dist_error)}")

#%% Normalize Adjacency Matrices
adj_train = utility.construct_normalized_adj(edge_train, x_train.shape[0])
adj_val = utility.construct_normalized_adj(edge_val, x_val.shape[0])
adj_test = utility.construct_normalized_adj(edge_test, x_test.shape[0])

#%% Creating dummy masks for PPI (because train, val, test is separate by default)
val_mask = np.full((x_val.shape[0],),True)
test_mask = np.full((x_test.shape[0],),True)

#%% Convert feature and labels to torch tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
val_mask = torch.tensor(val_mask)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_mask = torch.tensor(test_mask)

#%% Preparing Model
if (model_type == 'SIGN' or model_type == 'QSIGN'):
    # precomputing SIGN aggregation
    print("Pre-aggregating embeddings ...")
    x_train_aggregated = models.precompute_SIGN_aggregation(x_train, adj_train, 
                                                            max_hop)
    x_val_aggregated = models.precompute_SIGN_aggregation(x_val, adj_val, 
                                                            max_hop)
    x_test_aggregated = models.precompute_SIGN_aggregation(x_test, adj_test, 
                                                            max_hop)
    print(f"Pre-aggregated embedding size: {x_train_aggregated.shape[1]}")
    
    if model_type == 'QSIGN':
        model = models.QMLP(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= x_train_aggregated.shape[1],
                            out_channels=num_classes, batch_norm=batch_norm, 
                            dropout=dropout)
    elif model_type == 'SIGN':
        model = models.MLP(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= x_train_aggregated.shape[1],
                            out_channels=num_classes, batch_norm=batch_norm, 
                            dropout=dropout)
        
elif (model_type == 'GCN'):
    model = models.GCN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= x_train.shape[1], out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual)
    
print(model)

#%% Training
if model_type == 'SIGN' or model_type == 'QSIGN':
    if do_evaluation:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_SIGN.train(model, device, 
                                                          x_train=x_train_aggregated, 
                                                          y_train=y_train, 
                                                          x_val=x_val_aggregated, 
                                                          y_val=y_val, 
                                                          x_test=x_test_aggregated, 
                                                          y_test=y_test, 
                                                          multilabel=multilabel, 
                                                          lr=lr, num_batch=num_batch, 
                                                          num_epoch=num_epoch)
    else:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_SIGN.train(model, device, 
                                                          x_train=x_train_aggregated, 
                                                          y_train=y_train, 
                                                          x_val=None, 
                                                          y_val=None, 
                                                          x_test=None, 
                                                          y_test=None, 
                                                          multilabel=multilabel, 
                                                          lr=lr, num_batch=num_batch, 
                                                          num_epoch=num_epoch)

elif model_type == 'GCN':
    if do_evaluation:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_GCN.train(model, device, 
                                                          x_train=x_train, 
                                                          y_train=y_train, 
                                                          adj_train=adj_train,
                                                          x_val=x_val, 
                                                          y_val=y_val, 
                                                          adj_val=adj_val,
                                                          val_mask=val_mask,
                                                          x_test=x_test, 
                                                          y_test=y_test, 
                                                          adj_test=adj_test,
                                                          test_mask=test_mask,
                                                          multilabel=multilabel, 
                                                          lr=lr, num_epoch=num_epoch)
    else:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_GCN.train(model, device, 
                                                          x_train=x_train, 
                                                          y_train=y_train, 
                                                          adj_train=adj_train,
                                                          x_val=None, 
                                                          y_val=None, 
                                                          adj_val=None,
                                                          val_mask=None,
                                                          x_test=None, 
                                                          y_test=None, 
                                                          adj_test=None,
                                                          test_mask=None,
                                                          multilabel=multilabel, 
                                                          lr=lr, num_epoch=num_epoch)
    
#%% Printing Result
if print_result:
    writer.write_result(dataset_name, model_type, centrality, G_train.number_of_nodes(), 
                        max_val_acc, max_val_f1, max_val_sens, max_val_spec, 
                        max_val_test_acc, max_val_test_f1, max_val_test_sens, 
                        max_val_test_spec, session_memory, train_memory, 
                        train_time_avg)
    
    writer.write_label_dist_error(dataset_name, centrality, G_train.number_of_nodes(), 
                                  Y_dist_error)