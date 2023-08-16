#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:56:46 2023

@author: chris
"""
import numpy as np
import networkx as nx
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
    
def contract_graph(G, num_features, num_classes, 
                   centrality = 'EC', steps = [], 
                   multilabel = True):
    print("Contracting Graph...")
    print("=============== Graph Contraction ===============")
    print(f"Centrality Measure: {centrality}")
    print(f"Contraction steps: {steps}")
    print(f"Pre-contraction graph nodes: {G.number_of_nodes()}")
    print(f"Pre-contraction graph edges: {G.number_of_edges()}")
    
    random.seed(42) #42
        
    # Hierarchical Contraction
    for budget in steps:
        
        if centrality == 'BC':
            BC = nx.betweenness_centrality(G, k=100, normalized=True, weight=None, endpoints=False, seed=42)
        elif centrality == 'DC':
            BC = nx.degree_centrality(G)
        elif centrality == 'PR':
            BC = nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, dangling=None)
        elif centrality == 'EC':
            BC = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-06, nstart=None, weight=None)
        elif centrality == 'CC':
            BC = nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)
        elif centrality == 'RAND':
            BC = {}
            for i in range(G.number_of_nodes()):
                BC[i] = random.random()
        else:
            print("Centrality Type Undefined, No Contraction Done!")
            continue

        parts = np.zeros((G.number_of_nodes(),), dtype='int') # all nodes belong to the same partition
        
        # Reduce Training Graph (by nodes)
        N_budget = budget # Budget for number of training nodes
        num_training = 0
        for node in G.nodes(data=True):
            if node[1]['train']:
                num_training += 1
        remove_ratio = 1 - N_budget/num_training
    
        partitioned_num_nodes = np.zeros((1,))
        for node in G.nodes(data=True):
            if node[1]['train']:
                partitioned_num_nodes[parts[node[0]]] += 1
        
        N_remove_partition = np.zeros((1,))
        for i in range(1):
            N_remove_partition[i] = remove_ratio*partitioned_num_nodes[i]
        
        listBC = list(BC.items())
        random.shuffle(listBC)
        sortedBC = sorted(listBC, key=lambda x:x[1])
        
        for i in range(G.number_of_nodes()):
            
            # check the partition for node i
            curr_node = sortedBC[i][0]
            curr_part = parts[curr_node]
            
            # check if training node
            if not G.nodes[curr_node]['train']:
                continue
            
            # check the partition quota
            if (N_remove_partition[curr_part] <= 0.0):
                continue
            elif (random.random() > N_remove_partition[curr_part]):
                N_remove_partition[curr_part] -= 1
                continue
            
            N_remove_partition[curr_part] -= 1
            node_to_remove = curr_node
            neighbors = G.neighbors(node_to_remove)
            
            max_nb_BC = 0
            best_nb = None
            for nb in neighbors:
                if max_nb_BC < BC[nb]:
                    best_nb = nb
                    max_nb_BC = BC[nb]
        
            if best_nb != None: # remove node by merging with best neighbour
                nx.contracted_nodes(G, best_nb, node_to_remove, self_loops=False, copy=False)
            else: # if no neighbor, just remove self
                G.remove_node(node_to_remove)
            
        print(f"Post-contraction graph nodes: {G.number_of_nodes()}")
        print(f"Post-contraction graph edges: {G.number_of_edges()}")
        print()
            
    # Cleaning indexing after removal
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    
    return G
    