import os, time, sys
import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from iRevNet import iRevNetUtils

torch.set_default_dtype(torch.float32)

class binary(nn.Module):
    def __init__(self, chNum):
        super(binary, self).__init__()
        
    def forward(self, x):
        device = x.device
        bs, ch, T = x.shape
        F0 = ch//2
        F1 = ch-F0
        mask0 = torch.zeros((bs,F0,T)).cuda(device)
        mask1 = torch.ones((bs,F1,T)).cuda(device)
        mask = torch.cat((mask0, mask1), 1)
        return mask
    
    
class UNet5Sigmoid(nn.Module):
    def __init__(self, chNum):
        super(UNet5Sigmoid, self).__init__()
        self.kernelSize = (5, 3)
        self.pad = (2, 1)
        self.hidCh = 45
        self.e1 = nn.utils.spectral_norm(
                nn.Conv2d(1, 
                          self.hidCh, 
                          self.kernelSize,
                          stride = (1, 1),
                          padding = self.pad))
        self.e2 = nn.utils.spectral_norm(
                nn.Conv2d(self.hidCh, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad))
        self.e3 = nn.utils.spectral_norm(
                nn.Conv2d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad))
        self.e4 = nn.utils.spectral_norm(
                nn.Conv2d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad))
        self.e5 = nn.utils.spectral_norm(
                nn.Conv2d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad))
        
        self.d5 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(self.hidCh*2, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1))
        self.d4 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(self.hidCh*4, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1))
        self.d3 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(self.hidCh*4, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1))
        self.d2 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(self.hidCh*4, 
                                   self.hidCh, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1))
        self.d1 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(self.hidCh*2, 
                                   1, 
                                   self.kernelSize,
                                   stride = (1, 1),
                                   padding = self.pad))
        
    def forward(self, x):
        bs, ch, T = x.shape
        h0 = x.unsqueeze(1)
        
        h1 = nn.LeakyReLU(0.2)(self.e1(h0))
        h2 = nn.LeakyReLU(0.2)(self.e2(h1))
        h3 = nn.LeakyReLU(0.2)(self.e3(h2))
        h4 = nn.LeakyReLU(0.2)(self.e4(h3))
        h5 = nn.LeakyReLU(0.2)(self.e5(h4))
        
        dh = nn.LeakyReLU(0.2)(self.d5(h5))
        dh = torch.cat((dh, h4), 1)
        dh = nn.LeakyReLU(0.2)(self.d4(dh))
        dh = torch.cat((dh, h3), 1)
        dh = nn.LeakyReLU(0.2)(self.d3(dh))
        dh = torch.cat((dh, h2), 1)
        dh = nn.LeakyReLU(0.2)(self.d2(dh))
        dh = torch.cat((dh, h1), 1)
        dh = nn.Sigmoid()(self.d1(dh))
        mask = dh.reshape(x.shape)
        return mask
    
class insNormUNet5Sigmoid(nn.Module):
    def __init__(self, chNum):
        super(insNormUNet5Sigmoid, self).__init__()
        self.kernelSize = (5, 3)
        self.pad = (2, 1)
        self.hidCh = 45
        self.e1 = nn.Conv2d(1, 
                            self.hidCh, 
                            self.kernelSize,
                            stride = (1, 1),
                            padding = self.pad)
        self.INe1 = nn.InstanceNorm2d(self.hidCh)
        self.e2 = nn.Conv2d(self.hidCh, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad)
        self.INe2 = nn.InstanceNorm2d(self.hidCh*2)
        self.e3 = nn.Conv2d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad)
        self.INe3 = nn.InstanceNorm2d(self.hidCh*2)
        self.e4 = nn.Conv2d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad)
        self.INe4 = nn.InstanceNorm2d(self.hidCh*2)
        self.e5 = nn.Conv2d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = (2, 2),
                          padding = self.pad)
        self.INe5 = nn.InstanceNorm2d(self.hidCh*2)
        
        self.d5 = nn.ConvTranspose2d(self.hidCh*2, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1)
        self.INd5 = nn.InstanceNorm2d(self.hidCh*2)
        self.d4 = nn.ConvTranspose2d(self.hidCh*4, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1)
        self.INd4 = nn.InstanceNorm2d(self.hidCh*2)
        self.d3 = nn.ConvTranspose2d(self.hidCh*4, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1)
        self.INd3 = nn.InstanceNorm2d(self.hidCh*2)
        self.d2 = nn.ConvTranspose2d(self.hidCh*4, 
                                   self.hidCh, 
                                   self.kernelSize,
                                   stride = (2, 2),
                                   padding = self.pad,
                                   output_padding = 1)
        self.INd2 = nn.InstanceNorm2d(self.hidCh*2)
        self.d1 = nn.ConvTranspose2d(self.hidCh*2, 
                                   1, 
                                   self.kernelSize,
                                   stride = (1, 1),
                                   padding = self.pad)
         
        
    def forward(self, x):
        bs, ch, T = x.shape
        h0 = x.unsqueeze(1)
        
        h1 = nn.LeakyReLU(0.2)(self.INe1(self.e1(h0)))
        h2 = nn.LeakyReLU(0.2)(self.INe1(self.e2(h1)))
        h3 = nn.LeakyReLU(0.2)(self.INe1(self.e3(h2)))
        h4 = nn.LeakyReLU(0.2)(self.INe1(self.e4(h3)))
        h5 = nn.LeakyReLU(0.2)(self.INe1(self.e5(h4)))
        
        dh = nn.LeakyReLU(0.2)(self.INe1(self.d5(h5)))
        dh = torch.cat((dh, h4), 1)
        dh = nn.LeakyReLU(0.2)(self.INe1(self.d4(dh)))
        dh = torch.cat((dh, h3), 1)
        dh = nn.LeakyReLU(0.2)(self.INe1(self.d3(dh)))
        dh = torch.cat((dh, h2), 1)
        dh = nn.LeakyReLU(0.2)(self.INe1(self.d2(dh)))
        dh = torch.cat((dh, h1), 1)
        dh = nn.Sigmoid()(self.d1(dh))
        mask = dh.reshape(x.shape)
        return mask
    
