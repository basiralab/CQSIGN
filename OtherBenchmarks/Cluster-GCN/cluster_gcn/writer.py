#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:01:27 2023

@author: chris
"""
import os
import numpy as np

def create_empty_file_result(filename):
    f = open(filename, 'w')
    f.write('dataset,method,contraction,node_num,session_memory,train_memory,epoch_time,val_acc,val_f1,val_sens,val_spec,test_acc,test_f1,test_sens,test_spec\n')
    f.close()

def create_empty_file_label_dist(filename):
    f = open(filename, 'w')
    f.write('dataset,contraction,node_num,gamma,label_dist_avg_error,label_dist_error\n')
    f.close()

def write_result(dataset, model_type, centrality, node_num, max_val_acc, max_val_f1, 
                 max_val_sens, max_val_spec, max_val_test_acc, max_val_test_f1, 
                 max_val_test_sens, max_val_test_spec, session_memory, train_memory, 
                 train_time_avg, filename = "result.csv"):
    
    if not os.path.isfile(filename):
        create_empty_file_result(filename)
    
    f = open(filename, 'a')
    f.write(dataset)
    f.write(',')
    f.write(model_type)
    f.write(',')
    f.write(centrality)
    f.write(',')
    f.write(f"{node_num}")
    f.write(',')
    f.write(f"{session_memory}")
    f.write(',')
    f.write(f"{train_memory}")
    f.write(',')
    f.write(f"{train_time_avg}")
    f.write(',')
    f.write(f"{max_val_acc}")
    f.write(',')
    f.write(f"{max_val_f1}")
    f.write(',')
    f.write(f"{max_val_sens}")
    f.write(',')
    f.write(f"{max_val_spec}")
    f.write(',')
    f.write(f"{max_val_test_acc}")
    f.write(',')
    f.write(f"{max_val_test_f1}")
    f.write(',')
    f.write(f"{max_val_test_sens}")
    f.write(',')
    f.write(f"{max_val_test_spec}")
    f.write('\n')
    f.close()

def write_label_dist_error(dataset, centrality, node_num, gamma, Y_dist_error, 
                           filename = "label_distribution.csv"):
    
    if not os.path.isfile(filename):
        create_empty_file_label_dist(filename)
    
    f = open(filename, 'a')
    f.write(dataset)
    f.write(',')
    f.write(centrality)
    f.write(',')
    f.write(f"{node_num}")
    f.write(',')
    f.write(f"{gamma}")
    f.write(',')
    f.write(f"{np.mean(Y_dist_error)}")
    
    for ratio in Y_dist_error:
        f.write(',')
        f.write(f"{ratio}")
    f.write('\n')
    f.close()