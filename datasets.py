#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:55:25 2023

@author: chris
"""

from torch_geometric.datasets import Flickr
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import torch
import sklearn

def get_PPI(split = 'train'):
    print(f"Loading PPI ({split}) Dataset...")
    assert(split == 'train' or split == 'val' or split == 'test')
        
    data = PPI(root='data/PPI', split=split, transform=NormalizeFeatures())
    
    # Read Labels
    data_feat = data.x.numpy()
    
    # Read Features
    data_label = data.y.numpy()
    
    # Generate Mask
    train_mask = np.full((data_feat.shape[0],), False)
    val_mask = np.full((data_feat.shape[0],), False)
    test_mask = np.full((data_feat.shape[0],), False)
    if split == 'train':
        train_mask = np.full((data_feat.shape[0],), True)
    elif split == 'val':
        val_mask = np.full((data_feat.shape[0],), True)
    elif split == 'test':
        test_mask = np.full((data_feat.shape[0],), True)
        
    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    
    # Read Edges
    edge_index = data.edge_index
    edge_index = edge_index.numpy().T
    
    print_statistics(data_feat, data_label, edge_index, train_mask, val_mask, 
                     test_mask)
    
    return data_feat, data_label, edge_index, train_mask, val_mask, test_mask

def get_Organ(view = 'C'):
    print(f"Loading Organ-{view} Dataset...")
    
    if view == 'C':
        dataset_name = 'organc'
    elif view == 'S':
        dataset_name = 'organs'
    
    # Read Masks
    train_mask = np.load('data/'+dataset_name+'/train_mask.npy')
    train_mask = torch.tensor(train_mask)

    val_mask = np.load('data/'+dataset_name+'/val_mask.npy')
    val_mask = torch.tensor(val_mask)

    test_mask = np.load('data/'+dataset_name+'/test_mask.npy')
    test_mask = torch.tensor(test_mask)

    # Read Labels
    data_label = np.load('data/'+dataset_name+'/data_label.npy')

    # Read Features
    data_feat = np.load('data/'+dataset_name+'/data_feat.npy')
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(data_feat)
    data_feat = scaler.transform(data_feat)

    # Read Edges
    edge_index = np.load('data/'+dataset_name+'/edge_index.npy')
    
    print_statistics(data_feat, data_label, edge_index, train_mask, val_mask, 
                     test_mask)
    
    return data_feat, data_label, edge_index, train_mask, val_mask, test_mask

def print_statistics(data_feat, data_label, edge_index, train_mask, 
                     val_mask, test_mask):
    
    print("=============== Dataset Properties ===============")
    print(f"Total Nodes: {data_feat.shape[0]}")
    print(f"Total Edges: {edge_index.shape[0]}")
    print(f"Number of Features: {data_feat.shape[1]}")
    if len(data_label.shape) == 1:
        print(f"Number of Labels: {max(data_label) + 1}")
        print("Task Type: Multi-class Classification")
    else:
        print(f"Number of Labels: {data_label.shape[1]}")
        print("Task Type: Multi-label Classification")
    print(f"Training Nodes: {sum(train_mask)}")
    print(f"Validation Nodes: {sum(val_mask)}")
    print(f"Testing Nodes: {sum(test_mask)}")
    print()