
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import TimeSeriesModel
import pudb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_bins = 100 # Number of bins to classify price changes
bin_width = 0.002

n_lag = 120

model     = TimeSeriesModel(n_bins, n_lag).to(device)
model.eval()
checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(3500)) 
model.load_state_dict(checkpoint['state_dict'])


paths = get_txt_paths('dataset/valid/stocks')
fields = get_fields(paths[91], ['closing_prices'])
shift = 520
prices = torch.Tensor(fields['closing_prices'])[shift:]
n_values = fields['n_values']-shift


def get_samples(distrib):
    samples = torch.zeros(distrib.shape[0])
    for batch_idx in range(distrib.shape[0]):
        np.random.seed()
        sampled = False
        thresh = torch.max(distrib[batch_idx])*np.random.rand() + 1e-5
        n_draws = 0
        #samples[batch_idx] = (torch.argmax(distrib[batch_idx]) - n_bins // 2) * bin_width
        
        while not sampled:
            n_draws += 1
            idx = np.random.randint(0, n_bins)
            val = distrib[batch_idx, idx]
            if thresh < val or n_draws > 100:
                sampled = True
                samples[batch_idx] = (idx - n_bins // 2) * bin_width
    return samples

n_paths = 100
price_path = torch.zeros(n_paths, prices.shape[0])
price_path[:, :n_lag] = prices[:n_lag]
stop_idx = n_lag+160

for i in range(0, np.minimum(stop_idx, n_values-n_lag-1)):
    prices_in = price_path[:, i:i+n_lag]
    prices_pred = model(prices_in.unsqueeze(1).to(device)).detach()
    sample = get_samples(prices_pred)
    price_path[:, i+n_lag] = (1.+sample) * price_path[:, i+n_lag-1]

plt.figure()
for k in range(n_paths):
    plt.plot(price_path[k].cpu().numpy()[n_lag-1:stop_idx], c='b',alpha=0.1)
plt.plot(prices.numpy()[n_lag-1:stop_idx], c='g')
plt.plot(price_path[:,n_lag-1:stop_idx].cpu().numpy().mean(0), c='r', ls='--')
plt.show()
