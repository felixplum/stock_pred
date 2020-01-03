import numpy as np
import torch
import torch.nn as nn
import pudb

class TimeSeriesModel(nn.Module):
    def __init__(self, n_bins, n_lag):
        super (TimeSeriesModel, self).__init__() 
        self.n_bins = n_bins
        self.conv1 = nn.Sequential(nn.Conv1d(
                        in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=0), 
                        nn.ReLU(),
                        nn.MaxPool1d(2,2))
        self.conv2 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU(),
                        nn.MaxPool1d(2,2))
        self.conv3 = nn.Sequential(nn.Conv1d(
                        in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.Dropout(0.25),
                        nn.ReLU())
        
        self.out0 = nn.Sequential( nn.Linear(15*128, n_bins, bias = False),
                                  nn.Softmax(dim=1)
                                )
        self.out1 = nn.Sequential( nn.Linear(15*128, n_bins, bias = False),
                                  nn.Softmax(dim=1)
                                )
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        #x = self.conv7(x)
        # flatten
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #exit()
        out0 = self.out0(x)
        #out1 = self.out1(x)
        #print(x.shape)
        #exit()
        #x = self.conv2(x) 
        #print(out)
        return out0#, out1
    

'''
class TimeSeriesModel(nn.Module):
    def __init__(self, n_bins, n_lag):
        super (TimeSeriesModel, self).__init__() 
        self.n_bins = n_bins
        self.conv1 = nn.Sequential(nn.Conv1d(
                        in_channels=2, out_channels=128, kernel_size=5, stride=1, padding=0), 
                        nn.ReLU(),
                        nn.MaxPool1d(2,2))
        self.conv2 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU(),
                        nn.MaxPool1d(2,2))
        self.conv3 = nn.Sequential(nn.Conv1d(
                        in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv1d(
                        in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),                              
                        nn.ReLU())
        self.out0 = nn.Sequential( nn.Linear(15*128, n_bins, bias = False),
                                  nn.Softmax(dim=1)
                                )
        self.out1 = nn.Sequential( nn.Linear(15*128, n_bins, bias = False),
                                  nn.Softmax(dim=1)
                                )
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        #x = self.conv7(x)
        # flatten
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #exit()
        out0 = self.out0(x)
        out1 = self.out1(x)
        #print(x.shape)
        #exit()
        #x = self.conv2(x) 
        #print(out)
        return out0, out1
    
'''
