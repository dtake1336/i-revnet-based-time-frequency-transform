import os, time, sys
import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from iRevNet import iRevNetUtils
torch.set_default_dtype(torch.float32)

class UNet5SpecNorm(nn.Module):
    def __init__(self, inCh, outCh, decimate):
        super(UNet5SpecNorm, self).__init__()
        self.kernelSize = 5
        self.pad = self.kernelSize//2
        self.inCh = inCh
        self.outCh = outCh
        self.hidCh = 32
        self.e1 = nn.utils.spectral_norm(
                nn.Conv1d(self.inCh, 
                          self.hidCh, 
                          self.kernelSize,
                          stride = 1,
                          padding = self.pad))
        self.e2 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh, 
                          self.hidCh, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad))
        self.e3 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad))
        self.e4 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad))
        self.e5 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad))
        
        self.d5 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*2, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1))
        self.d4 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*4, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1))
        self.d3 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*4, 
                                   self.hidCh, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1))
        self.d2 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*2, 
                                   self.hidCh, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1))
        self.d1 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*2, 
                                   self.outCh, 
                                   self.kernelSize,
                                   stride = 1,
                                   padding = self.pad))
        
    def forward(self, x0):
        x1 =  nn.LeakyReLU(0.2)(self.e1(x0))
        x2 =  nn.LeakyReLU(0.2)(self.e2(x1))
        x3 =  nn.LeakyReLU(0.2)(self.e3(x2))
        x4 =  nn.LeakyReLU(0.2)(self.e4(x3))
        x5 =  nn.LeakyReLU(0.2)(self.e5(x4))
        
        dx =  nn.LeakyReLU(0.2)(self.d5(x5))
        dx = torch.cat((dx, x4), 1)
        dx =  nn.LeakyReLU(0.2)(self.d4(dx))
        dx = torch.cat((dx, x3), 1)
        dx =  nn.LeakyReLU(0.2)(self.d3(dx))
        dx = torch.cat((dx, x2), 1)
        dx =  nn.LeakyReLU(0.2)(self.d2(dx))
        dx = torch.cat((dx, x1), 1)
        dx =  nn.LeakyReLU(0.2)(self.d1(dx))
        return dx
    
    
class LinearNoBiasUNet5SpecNorm(nn.Module):
    def __init__(self, inCh, outCh, decimate):
        super(LinearNoBiasUNet5SpecNorm, self).__init__()
        self.kernelSize = 5
        self.pad = self.kernelSize//2
        self.inCh = inCh
        self.outCh = outCh
        self.hidCh = 32
        self.e1 = nn.utils.spectral_norm(
                nn.Conv1d(self.inCh, 
                          self.hidCh, 
                          self.kernelSize,
                          stride = 1,
                          padding = self.pad,
                          bias=False))
        self.e2 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh, 
                          self.hidCh, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad,
                          bias=False))
        self.e3 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad,
                          bias=False))
        self.e4 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad,
                          bias=False))
        self.e5 = nn.utils.spectral_norm(
                nn.Conv1d(self.hidCh*2, 
                          self.hidCh*2, 
                          self.kernelSize,
                          stride = 2,
                          padding = self.pad,
                          bias=False))
        
        self.d5 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*2, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1,
                                   bias=False))
        self.d4 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*4, 
                                   self.hidCh*2, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1,
                                   bias=False))
        self.d3 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*4, 
                                   self.hidCh, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1,
                                   bias=False))
        self.d2 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*2, 
                                   self.hidCh, 
                                   self.kernelSize,
                                   stride = 2,
                                   padding = self.pad,
                                   output_padding = 1,
                                   bias=False))
        self.d1 = nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.hidCh*2, 
                                   self.outCh, 
                                   self.kernelSize,
                                   stride = 1,
                                   padding = self.pad,
                                   bias=False))
        
    def forward(self, x0):
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        
        dx = self.d5(x5)
        dx = torch.cat((dx, x4), 1)
        dx = self.d4(dx)
        dx = torch.cat((dx, x3), 1)
        dx = self.d3(dx)
        dx = torch.cat((dx, x2), 1)
        dx = self.d2(dx)
        dx = torch.cat((dx, x1), 1)
        dx = self.d1(dx)
        return dx