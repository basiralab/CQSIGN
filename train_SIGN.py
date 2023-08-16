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

def mini_batch_step(model, optimizer, criterion, device, x_train, y_train, 
                         batch_size, logging = False):
    
    permutation = torch.randperm(x_train.shape[0])
    
    total_acc = 0
    total_micro_f1 = 0
    total_sens = 0
    total_spec = 0
    total_loss = 0
    
    for i in range(0,x_train.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        
        x_batch = x_train[indices].to(device)
        y_batch = y_train[indices].to(device)
        
        model.train()
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        total_loss += loss*indices.shape[0]
        loss.backward()
        optimizer.step()
    
        if logging:
            acc,micro_f1,sens,spec = utility.metrics(out,y_batch)
            total_acc += acc*indices.shape[0]
            total_micro_f1 += micro_f1*indices.shape[0]
            total_sens += sens*indices.shape[0]
            total_spec += spec*indices.shape[0]
        
        del x_batch
        del y_batch
            
    if logging:
        total_acc /= x_train.shape[0]
        total_micro_f1 /= x_train.shape[0]
        total_sens /= x_train.shape[0]
        total_spec /= x_train.shape[0]
        print(f"Train accuracy: {total_acc}, Train Micro_f1: {total_micro_f1} Train Sens: {total_sens}, Train Spec: {total_spec}")
        
    return total_loss/x_train.shape[0]

def evaluate(model, x, y):

    with torch.no_grad():
        model.eval()
        out = model(x)
        acc,micro_f1,sens,spec = utility.metrics(out,y)
    
    return acc, micro_f1, sens, spec


def train(model, device, x_train, y_train, x_val = None, y_val = None, 
                x_test = None, y_test = None, multilabel = True, 
                lr = 0.0005, num_batch = 10, num_epoch = 100):
    
    # passing model to GPU
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = math.ceil(x_train.shape[0]/num_batch)
    
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
        loss = mini_batch_step(model, optimizer, criterion, device, 
                                    x_train, y_train, batch_size, 
                                    logging = False)
        time_per_epoch = time.time() - t
        time_arr[epoch] = time_per_epoch
        
        if epoch == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
            
            # passing validation and test data to GPU (we do it after first forward pass to get)
            # accurate pure training GPU memory usage
            if x_val != None and y_val != None:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                if x_test != None and y_test != None:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}')
            print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
        
        # evaluation
        if x_val != None and y_val != None:
            acc, micro_f1, sens, spec = evaluate(model, x_val, y_val)
            
            if epoch % 100 == 0:
                print(f"Val accuracy: {acc}, Val micro_f1: {micro_f1}, Val Sens: {sens}, Val Spec: {spec}")
            
            if acc > max_val_acc:
                max_val_acc = acc
                max_val_f1 = micro_f1
                max_val_sens = sens
                max_val_spec = spec
                
                if (x_test != None and y_test != None):
                    acc, micro_f1, sens, spec = evaluate(model, x_test, y_test)
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
    
    
    