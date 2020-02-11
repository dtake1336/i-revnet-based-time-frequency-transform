import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def initSplit(x):
    x_o = (x[:,::2]).unsqueeze(1)
    x_e = (x[:,1::2]).unsqueeze(1)
    return x_o, x_e

def initSplitInv(x_o, x_e):
    bs, _, _ = x_o.shape
    x = torch.cat((x_o, x_e), 1)
    x = ch2space(x, 2)
    x = x.view(bs,-1)
    return x


def ch2space(x,stride):
    bs, chNum, picNum = x.shape
    x = x.reshape(bs, stride, chNum//stride, picNum)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(bs, chNum//stride, picNum*stride)
    return x
    
def space2ch(x, stride):
    bs, ch, T = x.shape
    x = x.view(bs, ch, T//stride, stride)
    x = x.permute(0, 3, 1, 2)
    x = x.reshape(bs, ch*stride, T//stride)
    return x


def iRevNetBlock_forward(x0a, x0b, Filter, decimate):
    Fx0b = Filter(x0b)
    x1a = x0b
    x1b = x0a+Fx0b
    if decimate != 1.:
        x1a = space2ch(x1a, decimate)
        x1b = space2ch(x1b, decimate)
    return x1a,x1b

def iRevNetBlock_backward(y1a,y1b,Filter,decimate):
    if decimate != 1.:
        y1a = ch2space(y1a,decimate)
        y1b = ch2space(y1b,decimate)
    Fy1a = Filter(y1a)
    y0a = y1b-Fy1a
    y0b = y1a
    return y0a,y0b