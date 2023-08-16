import json
import os
import os.path as osp
from typing import Callable, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import torch_geometric.transforms as T
import networkx as nx

def get_dataset(root: str, name: str, split: str) -> Tuple[Data, int, int]:
    
    dataset = read_data(f'{root}/{name}/{split}', pre_transform=T.ToSparseTensor())
    return dataset[0], dataset.num_features, dataset.num_classes

class read_data(InMemoryDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return 'no_data'

    @property
    def processed_file_names(self) -> str:
        return 'no_data' # force always process

    def download(self):
        pass

    def process(self):
        # Read Masks
        print("Processing Dataset...")
        train_mask = np.load(osp.join(self.raw_dir, 'train_mask.npy'))
        
        val_mask = np.load(osp.join(self.raw_dir, 'val_mask.npy'))
        
        test_mask = np.load(osp.join(self.raw_dir, 'test_mask.npy'))
        
        # Read Labels
        y = np.load(osp.join(self.raw_dir, 'data_label.npy'))
        
        # Read Features
        x = np.load(osp.join(self.raw_dir, 'data_feat.npy'))

        # Read Edges
        edge_index = np.load(osp.join(self.raw_dir, 'edge_index.npy'))
        
        #%%
        x = torch.tensor(x, dtype=torch.float32)
        if y.ndim == 1:
            y = torch.tensor(y) # for multiclass
        else:
            y = torch.tensor(y, dtype=torch.float32) # for multilabel
        edge_index = torch.tensor(edge_index)
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)
        
        #%%
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])