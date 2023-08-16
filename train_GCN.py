#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:01:29 2023

@author: chris
"""

import torch
import utility
import time
import numpy as np
import math

def full_batch_step(model, optimizer, criterion, device, x_train, y_train, 
                    adj_train, train_mask, logging = False):
        
    model.train()
    optimizer.zero_grad()
    out = model(x_train, adj_train)
    if train_mask == None:
        loss = criterion(out, y_train)
    else:
        loss = criterion(out[train_mask], y_train[train_mask])
    loss.backward()
    optimizer.step()

    if logging:
        acc,micro_f1,sens,spec = utility.metrics(out,y_train)
        print(f"Train accuracy: {acc}, Train micro_f1: {micro_f1},Train Sens: {sens}, Train Spec: {spec}")
        
    return loss

def evaluate(model, x, y, adj, mask):

    with torch.no_grad():
        model.eval()
        out = model(x, adj)
        acc,micro_f1,sens,spec = utility.metrics(out[mask],y[mask])
    
    return acc, micro_f1, sens, spec


def train(model, device, x_train, y_train, adj_train, train_mask = None, x_val = None, 
          y_val = None, adj_val = None, val_mask = None, x_test = None, 
          y_test = None, adj_test = None, test_mask = None, multilabel = True, 
          lr = 0.0005, num_batch = 10, num_epoch = 100):
    
    # passing model and training data to GPU
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    adj_train = adj_train.to(device)
    if train_mask != None:
        train_mask = train_mask.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    max_val_acc = 0
    max_val_sens = 0
    max_val_spec = 0
    max_val_f1 = 0
    max_val_test_acc = 0
    max_val_test_sens = 0
    max_val_test_spec = 0
    max_val_test_f1 = 0
    
    time_arr = np.zeros((num_epoch,))
    for epoch in range(num_epoch):
            
        # single mini batch step
        t = time.time()
        loss = full_batch_step(model, optimizer, criterion, device, 
                                x_train, y_train, adj_train, train_mask, 
                                logging = False)
        time_per_epoch = time.time() - t
        time_arr[epoch] = time_per_epoch
        
        if epoch == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
            
            # passing validation and test data to GPU (we do it after first forward pass to get)
            # accurate pure training GPU memory usage
            if x_val != None and y_val != None and adj_val != None and val_mask != None:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                adj_val = adj_val.to(device)
                val_mask = val_mask.to(device)
                if x_test != None and y_test != None and adj_test != None and test_mask != None:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    adj_test = adj_test.to(device)
                    test_mask = test_mask.to(device)
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}')
            print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
        
        # evaluation
        if x_val != None and y_val != None:
            acc, micro_f1, sens, spec = evaluate(model, x_val, y_val, adj_val, 
                                                 val_mask)
            
            if epoch % 100 == 0:
                print(f"Val accuracy: {acc}, Val micro_f1: {micro_f1}, Val Sens: {sens}, Val Spec: {spec}")
            
            if acc > max_val_acc:
                max_val_acc = acc
                max_val_f1 = micro_f1
                max_val_sens = sens
                max_val_spec = spec
                
                if (x_test != None and y_test != None):
                    acc, micro_f1, sens, spec = evaluate(model, x_test, y_test, 
                                                         adj_test, test_mask)
                    max_val_test_acc = acc
                    max_val_test_f1 = micro_f1
                    max_val_test_sens = sens
                    max_val_test_spec = spec
                    
                    print("===========================================Best Model Update:=======================================")
                    print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
                    print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
                    print("====================================================================================================")

    print("Best Model:")
    print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
    print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
    print(f"Average time per epoch: {time_arr[10:].mean()}") # don't include the first few epoch (slower due to Torch initialization)
    print(f"Training GPU Memory Usage: {train_memory} MB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
    
    # cleaning memory and stats
    session_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
    train_time_avg = time_arr[10:].mean()
    del x_val
    del y_val
    del x_test
    del y_test
    model = model.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    return (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc,
            max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
            train_memory, train_time_avg)
    
    
    