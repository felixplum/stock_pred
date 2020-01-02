import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from model import TimeSeriesModel
from dataloader import get_dataloader
import pudb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_epoch = 0
num_epochs  = 15000
learning_rate = 1e-4
n_bins = 100 # Number of bins to classify price changes
n_lag = 120

def main():

    model     = TimeSeriesModel(n_bins, n_lag).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

    if start_epoch != 0:
        checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(start_epoch-1)) 
        model.load_state_dict(checkpoint['state_dict'])
    
    data_loaders, data_size = get_dataloader(batch_size=128, n_lag=n_lag)
    for epoch in range(start_epoch, num_epochs + start_epoch):
        
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, num_epochs + start_epoch - 1))
        t0 = time.time()    
        loss = train_valid(model, optimizer, scheduler, epoch, data_loaders, data_size)
        #scheduler.step()
        t1 = time.time()
        #print("It took {0}s".format(t1-t0))
    print(80 * '=') 


def map_float_to_class_idx(values, n_bins):
    # map change values to 0..n_bins
    #resolution = 0.001 # 100 equidistant bins: 10% change possible
    range_max = 0.5 * n_bins * bin_width
    eps = 1e-6
    # TODO: finer resolution for more likely ( = smaller) values
    t0 = (torch.clamp(values, -range_max+eps, range_max-eps) + range_max) / (2.*range_max) # 0..0.99
    t1 = (t0 * n_bins).long() # convert to idx
    #print(values, t1)
    return t1

def map_float_to_normal_distr(values, n_bins, bin_width, T):
    # For each batch: Sample around given value
    # evaluate normal distr. at bin center:
    val_grid = ( (torch.arange(0, n_bins) - n_bins // 2) * bin_width).to(device)
    range_max = n_bins // 2 * bin_width
    output = torch.zeros(values.shape[0],n_bins).to(device)
    for batch_idx in range(values.shape[0]):
        prob = torch.exp(- (torch.clamp(values[batch_idx], -range_max, range_max)-val_grid)**2 / T  )
        prob /= (torch.sum(prob)+1e-9)
        output[batch_idx] = prob
    return output

def map_class_idx_to_float(values, n_bins):
    # map change values to 0..n_bins
    range_max = 0.5 * n_bins * bin_width - 1e-9
    return 2.*range_max*values / n_bins - range_max

def write_losses(losses):
    if not isinstance(losses, list):
        losses = [losses]
    with open('log/losses.txt','a+') as file:
        for l in losses:
            file.write(str(l) + '\n')
    file.close()


def get_kl_loss(P, Q):
    eps = 1e-9
    #return torch.sum((P-Q)**2)
    return ( (P+eps) * ( (P+eps)/(Q+eps) ).log()).sum()

def train_valid(model, optimizer, scheduler, epoch, dataloaders, data_size):
    
    for phase in ['train']: #['train', 'valid']:

        if phase == 'train':
            model.train()
        else:
            model.eval()
       
        loss_total = 0.
        n_samples  = 0
        plt.figure(0)
        plt.ion()
        
        for batch_idx, batch_sample in enumerate(dataloaders[phase]):
            
            # Inputs
            input_prices  = batch_sample['last_prices'].unsqueeze(1).to(device)  + 1e-9
            input_volumes = batch_sample['last_volumes'].unsqueeze(1).to(device) + 1e-9
            input_all  = torch.cat((input_prices, input_volumes), dim=1)
            # Output: Express next price as percentage change from last observed price
            next_prices         = batch_sample['next_price'].unsqueeze(1).to(device)
            price_change_target = (next_prices - input_prices[:,:, -1]) / input_prices[:,:, -1]
            price_change_distr_target = map_float_to_normal_distr(price_change_target, n_bins, bin_width=0.2/n_bins, T=1e-5)
            
            next_volume          = batch_sample['next_volume'].unsqueeze(1).to(device)
            volume_change_target = (next_volume - input_volumes[:,:, -1]) / input_volumes[:,:, -1]
            #plt.hist(volume_change_target.detach().cpu().numpy(), 100)
            #plt.hist(input_volumes[0, 0].detach().cpu().numpy(), 100)
            #plt.show()
            volume_change_distr_target = map_float_to_normal_distr(volume_change_target, n_bins, \
                                                                   bin_width=2./n_bins, T=1e-5)

            with torch.set_grad_enabled(phase == 'train'):
                
                # compute distribution for percentage deviation from last day in input sequence
                price_predict, volume_predict = model(input_all)#.to(device)
                plt.clf() 
                plt.plot(price_predict[0].detach().cpu().numpy(), 'r')
                plt.plot(price_change_distr_target[0].detach().cpu().numpy(), 'g')
                plt.ylim([0,1])
                #plt.plot(volume_predict[0].detach().cpu().numpy(), 'r')
                #plt.plot(volume_change_distr_target[0].detach().cpu().numpy(), 'g')
                plt.show()
                plt.pause(0.001)
                #print(volume_change_distr_target[0].detach().cpu().numpy())
                # Compute Kullback-Leibler divergence, i.e. "distance" between two distributions
                kl_loss     = get_kl_loss(price_change_distr_target,  price_predict) + \
                              get_kl_loss(volume_change_distr_target, volume_predict)
                n_samples  += input_prices.shape[0]
                loss_total += kl_loss.detach().item()
                
                if phase == 'train':
                    optimizer.zero_grad()
                    kl_loss  .backward()
                    optimizer.step()
        
        loss_mean = loss_total/n_samples
        write_losses(loss_mean)
        print("loss in epoch {0}: {1}".format(epoch, loss_mean))
        if phase == 'train' and epoch % 50 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                        './log/checkpoint_epoch{}.pth'.format(epoch))
        return loss_mean
if __name__ == '__main__':
    
    main()
