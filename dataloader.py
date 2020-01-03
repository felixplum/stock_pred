import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pudb
import csv
from utils import get_txt_paths, get_fields
class StockDataset():


    def __init__(self, root_path, n_lag):
	# load file list, store paths
        self.paths = get_txt_paths(root_path)
        self.n_lag = n_lag

    def z_normalize(self, batch_data):
        tmp  = torch.log(batch_data+1)
        n_samples = batch_data.shape[0]
        mean      = torch.mean(tmp)
        diff      = tmp - mean
        std       = torch.sqrt(torch.sum(torch.mul(diff, diff)) / n_samples) + 1e-9
        return (diff) / (3*std)

    def __getitem__(self, idx):
        # returns sequence of n_lag values and 1 value after (which is to be predicted)
        fields = get_fields(self.paths[idx], ['closing_prices', 'volume'])
        prices = torch.Tensor(fields['closing_prices'])
        volume = torch.Tensor(fields['volume'])
        n_values = fields['n_values']
        
        def get_io(n_lag):
            if n_values-n_lag < 1:
                return None, None, None, None
            np.random.seed()
            #start_idx = 100*np.random.randint(2) # TODO test only
            start_idx = np.random.randint(0, n_values-n_lag)
            stop_idx  = start_idx + n_lag
            return prices[start_idx:stop_idx], prices[stop_idx], volume[start_idx:stop_idx], volume[stop_idx]
        
        last_prices, next_price, last_volumes, next_volume = get_io(self.n_lag)
        
        sample = {'last_prices': last_prices, 'next_price':next_price, \
                  'last_volumes': last_volumes, 'next_volume' : next_volume}
        
        return sample
    
    
    def __len__(self):    
        return len(self.paths)
 
def my_collate(batch):
    batch = list(filter (lambda x:x['last_prices'] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_dataloader(batch_size, n_lag):
    # load datasets
    num_workers = 16
    stock_dataset = {'train': StockDataset('dataset/train/stocks/', n_lag), 'valid': StockDataset('dataset/valid/stocks', n_lag)}
    dataloaders = {
        x: torch.utils.data.DataLoader(stock_dataset[x], batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn=my_collate)
        for x in ['train', 'valid']}
    
    data_size = {x: len(stock_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size

