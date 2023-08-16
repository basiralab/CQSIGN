#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:10:24 2023

@author: chris
"""

import torch
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor
from layers import QLinear, QBatchNorm1d, QDropout, QReLU
import sys

seed = 1 #1

def precompute_SIGN_aggregation(x, adj, max_hop):
    x_hop = x
    x_aggregated = x_hop
    for i in range(max_hop):
        x_hop = adj @ x_hop
        x_aggregated = torch.cat((x_aggregated, x_hop),1)
    
    return x_aggregated

# MLP model
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm = False, dropout = 0.0, 
                 drop_input = False):
        
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        
        self.linear_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        
        # Adding input layer
        self.linear_layers.append(torch.nn.Linear(in_channels, hidden_channels))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding hidden layers
        for i in range(num_layers-2):
            self.linear_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding output layer
        self.linear_layers.append(torch.nn.Linear(hidden_channels, out_channels))
    
    def forward(self, x):
        
        # if using input dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        for i in range(self.num_layers-1): # exclude output layer
            x = self.linear_layers[i](x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.linear_layers[-1](x) # output layer
        
        return x


# Quantized MLP model
class QMLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm = False, dropout = 0.0, 
                 drop_input = False):
        
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        
        self.linear_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        self.relu_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        
        if drop_input:
            self.input_dropout_layer = QDropout(self.dropout)
        
        # Adding input layer
        self.linear_layers.append(QLinear(in_channels, hidden_channels))
        self.relu_layers.append(QReLU())
        self.dropout_layers.append(QDropout(self.dropout))
        if self.batch_norm:
            self.batch_norm_layers.append(QBatchNorm1d(hidden_channels))
        
        # Adding hidden layers
        for i in range(num_layers-2):
            self.linear_layers.append(QLinear(hidden_channels, hidden_channels))
            self.relu_layers.append(QReLU())
            self.dropout_layers.append(QDropout(self.dropout))
            if self.batch_norm:
                self.batch_norm_layers.append(QBatchNorm1d(hidden_channels))
        
        # Adding output layer
        self.linear_layers.append(QLinear(hidden_channels, out_channels))
    
    def forward(self, x):
        
        # if using input dropout
        if self.drop_input:
            x = self.input_dropout_layer(x)
            
        for i in range(self.num_layers-1): # exclude output layer
            x = self.linear_layers[i](x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = self.relu_layers[i](x)
            x = self.dropout_layers[i](x)
        
        x = self.linear_layers[-1](x) # output layer
        
        return x
        
# SIGN Model
class Original_SIGN(torch.nn.Module):
    def __init__(self, single_scale_hidden_channels, MLP_hidden_channels, 
                 MLP_num_layers, single_scale_in_channels, num_scales, out_channels, 
                 batch_norm = False, dropout = 0.0, drop_input = False):
        
        super().__init__()
        torch.manual_seed(seed)
        self.MLP_hidden_channels = MLP_hidden_channels
        self.single_scale_hidden_channels = single_scale_hidden_channels
        self.MLP_num_layers = MLP_num_layers
        self.single_scale_in_channels = single_scale_in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        self.num_scales = num_scales
        
        self.single_scale_linear_layers = torch.nn.ModuleList()
        self.single_scale_batch_norm_layers = torch.nn.ModuleList()
        
        # Adding input group layers
        for i in range(self.num_scales):
            self.single_scale_linear_layers.append(torch.nn.Linear(single_scale_in_channels, single_scale_hidden_channels))
            if self.batch_norm:
                self.single_scale_batch_norm_layers.append(torch.nn.BatchNorm1d(single_scale_hidden_channels))
        
        # Adding MLP layers
        self.MLP = MLP(hidden_channels=MLP_hidden_channels, num_layers=MLP_num_layers,
                       in_channels=num_scales*single_scale_hidden_channels,
                       out_channels=out_channels, batch_norm=batch_norm,
                       dropout=dropout, drop_input=drop_input)
    
    def forward(self, x_scales):
        assert(x_scales.shape[0] == self.num_scales)
        
        # Scale input layer
        x = torch.zeros((x_scales.shape[1],0)).to(x_scales.get_device())
        for i in range(self.num_scales):
            # if using input dropout
            if self.drop_input:
                x_scales[i] = F.dropout(x_scales[i], p=self.dropout, training=self.training)
            
            h = self.single_scale_linear_layers[i](x_scales[i])
            if self.batch_norm:
                h = self.single_scale_batch_norm_layers[i](h)
            x = torch.cat((x, h),1)
        
        x = x.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # MLP layers
        x = self.MLP(x)
        
        return x
            
# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm = False, dropout = 0.0, 
                 drop_input = False, residual = False):
        
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        self.residual = residual
        
        self.linear_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
            
        # Adding input layer
        if residual:
            self.linear_layers.append(torch.nn.Linear(2*in_channels, hidden_channels))
        else:
            self.linear_layers.append(torch.nn.Linear(in_channels, hidden_channels))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Adding hidden layers
        for i in range(num_layers-2):
            if residual:
                self.linear_layers.append(torch.nn.Linear(2*hidden_channels, hidden_channels))
            else:
                self.linear_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding output layer
        if residual:
            self.linear_layers.append(torch.nn.Linear(2*hidden_channels, out_channels))
        else:
            self.linear_layers.append(torch.nn.Linear(hidden_channels, out_channels))
    
    def forward(self, x, adj):
        
        # if using input dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        for i in range(self.num_layers-1): # exclude output 
            # aggregation phase
            if self.residual:
                x = torch.cat((adj @ x,x),1)
            else:
                x = adj @ x
            
            # transformation phase
            x = self.linear_layers[i](x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # aggregation phase (output layer)
        if self.residual:
            x = torch.cat((adj @ x,x),1)
        else:
            x = adj @ x
        
        # transformation phase (output layer)
        x = self.linear_layers[-1](x)
        
        return x
            
            
            
        
        
        
        