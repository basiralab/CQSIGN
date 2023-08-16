#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 00:44:09 2023

@author: chris
"""

import numpy as np
from sklearn import metrics as sk

views = ['c', 's']

for view in views:
    
    num_features = 28*28
    
    # Processing Dataset
    train_images = np.load("data/organ" + view + "/train_images.npy")
    train_data = train_images.reshape(train_images.shape[0],num_features).astype('float')
    train_labels = np.load("data/organ" + view + "/train_labels.npy")
    num_train = train_images.shape[0]
    
    val_images = np.load("data/organ" + view + "/val_images.npy")
    val_data = val_images.reshape(val_images.shape[0],num_features).astype('float')
    val_labels = np.load("data/organ" + view + "/val_labels.npy")
    num_val = val_images.shape[0]
    
    test_images = np.load("data/organ" + view + "/test_images.npy")
    test_data = test_images.reshape(test_images.shape[0],num_features).astype('float')
    test_labels = np.load("data/organ" + view + "/test_labels.npy")
    num_test = test_images.shape[0]
    
    # Concatenate train, validation, and test data
    num_data = num_train + num_val + num_test
    data_feat = np.concatenate((train_data, val_data, test_data), axis = 0)
    data_label = np.concatenate((train_labels, val_labels, test_labels), axis = 0)
    data_label = data_label.reshape((data_label.shape[0],))
    
    # Construct Adjacency Matrix
    A = sk.pairwise.cosine_similarity(data_feat,data_feat)
    
    # Scaled Adjacency Matrix
    maxA = np.max(A)
    minA = np.min(A)
    A = (A - minA)/(maxA - minA)
    
    # Sparsifying Adjacency Using Threshold (Number is set such that it 
    # results in graph with ~1.2 million edges)
    if view == 'c': # organ-c
        A = A > 0.972
    elif view == 's': # organ-s
        A = A > 0.977
    
    # Generating Masks
    train_mask = np.full((num_data),False)
    train_mask[:num_train] = True
    np.save("data/organ" + view + '/train_mask.npy', train_mask)
    
    val_mask = np.full((num_data),False)
    val_mask[num_train:num_train+num_val] = True
    np.save("data/organ" + view + '/val_mask.npy', val_mask)
    
    test_mask = np.full((num_data),False)
    test_mask[num_train+num_val:] = True
    np.save("data/organ" + view + '/test_mask.npy', test_mask)
    
    # generate feature tensor
    np.save("data/organ" + view + '/data_feat.npy', data_feat)
    
    # generate label tensor
    np.save("data/organ" + view + '/data_label.npy', data_label)
    
    # generate edge index
    edge_index = []
    for i in range(num_data):
        for j in range(num_data):
            if (i != j and A[i][j] == True):
                edge_index.append([i,j])
                
    np.save("data/organ" + view + '/edge_index.npy', edge_index)
    print(f"view-{view} generated!")

