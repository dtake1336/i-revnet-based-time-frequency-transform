import torch
import torch.nn as nn
import torch.nn.functional as F

def SDR(s, x, y, thres=20):
    n = x - s
    d = x - y
    diff = s - y
    sSDR = 10.0 * torch.log10( (s**2.0).sum(1) / ((diff**2.0).sum(1) + 1e-8) )
    diff = n - d
    nSDR = 10.0 * torch.log10( (n**2.0).sum(1) / ((diff**2.0).sum(1) + 1e-8) )
    loss = 0.5 * (-thres * torch.tanh( sSDR/thres ) - thres * torch.tanh( nSDR/thres ) )
    return loss.mean()