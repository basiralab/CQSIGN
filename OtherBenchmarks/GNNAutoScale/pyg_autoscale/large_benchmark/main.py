import time
import hydra
from omegaconf import OmegaConf
from read_dataset import get_dataset
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, dropout)
from sklearn.metrics import confusion_matrix
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Optional, Tuple
import sys
import writer
import argparse

torch.manual_seed(42) #42

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

def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None,
               edge_dropout=0.0):
    model.train()

    total_loss = total_examples = 0
    for i, (batch, batch_size, *args) in enumerate(loader):
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)
    
        if train_mask.sum() == 0:
            continue

        # We make use of edge dropout on ogbn-products to avoid overfitting.
        adj_t = dropout(adj_t, p=edge_dropout)

        optimizer.zero_grad()
        out = model(x, adj_t, batch_size, *args)
        
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        # We may abort after a fixed number of steps to refresh histories...
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break
        
    return total_loss / total_examples


@torch.no_grad()
def full_test(model, data):
    model.eval()
    return model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()

@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    return model(loader=loader)

@hydra.main(config_path='conf', config_name='config')
def main(conf):
    
    dataset_centrality = conf.dataset_centrality
    num_epoch = conf.num_epoch
    config_dataset = conf.dataset_name
    do_eval = conf.do_eval
    device = conf.device
    
    conf.model.params = conf.model.params[config_dataset]
    params = conf.model.params
    print(params)
    print(OmegaConf.to_yaml(conf))
    try:
        edge_dropout = params.edge_dropout
    except:  # noqa
        edge_dropout = 0.0
    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    t = time.perf_counter()
    print('Loading data...', end=' ', flush=True)
    
    data, in_channels, out_channels = get_dataset('data',dataset_centrality,'train')
    val_data, _, _ = get_dataset('data',dataset_centrality,'val')
    test_data, _, _ = get_dataset('data',dataset_centrality,'test')
    
    print(f"in_channels: {in_channels}")
    print(f"out_channels: {out_channels}")
    print(f"Number of training nodes: {data.x.shape}")
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    if conf.model.loop:
        data.adj_t = data.adj_t.set_diag()
        val_data.adj_t = val_data.adj_t.set_diag()
        test_data.adj_t = test_data.adj_t.set_diag()
    if conf.model.norm:
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        val_data.adj_t = gcn_norm(val_data.adj_t, add_self_loops=False)
        test_data.adj_t = gcn_norm(test_data.adj_t, add_self_loops=False)

    if data.y.dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0)

    eval_loader = EvalSubgraphLoader(data, ptr,
                                     batch_size=params['batch_size'])
        
    t = time.perf_counter()
    print('Calculating buffer size...', end=' ', flush=True)
    # We reserve a much larger buffer size than what is actually needed for
    # training in order to perform efficient history accesses during inference.
    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader])
    print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')

    kwargs = {}
    GNN = getattr(models, conf.model.name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=params.pool_size,
        buffer_size=buffer_size,
        **params.architecture,
        **kwargs,
    ).to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(),
             weight_decay=params.reg_weight_decay),
        dict(params=model.nonreg_modules.parameters(),
             weight_decay=params.nonreg_weight_decay)
    ], lr=params.lr)

    t = time.perf_counter()
    print('Fill history...', end=' ', flush=True)
    mini_test(model, eval_loader)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    total_training_time = 0.0
    
    max_val_acc = 0
    max_val_sens = 0
    max_val_spec = 0
    max_val_f1 = 0
    
    max_val_test_acc = 0
    max_val_test_sens = 0
    max_val_test_spec = 0
    max_val_test_f1 = 0
    
    for epoch in range(num_epoch):
        t = time.time()

        loss = mini_train(model, train_loader, criterion, optimizer,
                          params.max_steps, grad_norm, edge_dropout)

        dt = time.time() - t
        total_training_time += dt
        
        if epoch == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
        
        session_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, total training time: {total_training_time:.5f}')
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")

        if do_eval:
            
            out_val = full_test(model, val_data)
            
            val_acc, val_f1, val_sens, val_spec = metrics(out_val[val_data.val_mask], val_data.y[val_data.val_mask])
            
            print(f"Val accuracy: {val_acc}, Val micro_f1: {val_f1}, Val Sens: {val_sens}, Val Spec: {val_spec}")
            
            if (val_acc > max_val_acc):
                
                max_val_acc = val_acc
                max_val_f1 = val_f1
                max_val_sens = val_sens
                max_val_spec = val_spec
            
                out_test = full_test(model, test_data)
                
                test_acc, test_f1, test_sens, test_spec = metrics(out_test[test_data.test_mask], test_data.y[test_data.test_mask])
                
                max_val_test_acc = test_acc
                max_val_test_f1 = test_f1
                max_val_test_sens = test_sens
                max_val_test_spec = test_spec
                
                print("===========================================Best Model Update:=======================================")
                print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
                print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
                print("====================================================================================================")
    
    train_time_avg = total_training_time/num_epoch
    
    writer.write_result(dataset_centrality, 'GNNAutoScale', dataset_centrality, data.x.shape[0], max_val_acc, 
                        max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, max_val_test_f1, 
                        max_val_test_sens, max_val_test_spec, session_memory, train_memory, 
                        train_time_avg)

if __name__ == "__main__":
    sys.argv.append('hydra.job.chdir=False')
    main()
