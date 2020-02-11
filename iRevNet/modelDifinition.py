import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from iRevNet import iRevNetMaskEstimator, iRevNetFilter, iRevNetUtils


class iRevNetMasking(nn.Module):
    def __init__(self, layerNum, filt, initPad, maskEstimator):
        super(iRevNetMasking, self).__init__()
        self.initPad = initPad
        self.chNum = [1+initPad]
        self.layerNum = layerNum
        for ii in range(1,self.layerNum+1):
            self.chNum.append(self.chNum[0]*(2**ii))
        self.decimate = ( ((np.array(self.chNum[1:])) / np.array(self.chNum[:-1])).astype(np.int) ).tolist()
        
        self.maskEst = eval('iRevNetMaskEstimator.'+maskEstimator+'(self.chNum[-1])')
        
        for layer in range(self.layerNum):
            tmptxt = "self.Filt"+str(layer+1)+" = iRevNetFilter."+filt+"("\
                 +"self.chNum["+str(layer)+"], "\
                 +"self.chNum["+str(layer)+"], "\
                 +"self.decimate["+str(layer)+"])"
            exec(tmptxt)
            
    def forward(self, x):
        device = x.device
        xaInit, xbInit = iRevNetUtils.initSplit(x)
        bs, _, T = xaInit.shape
        tmpPad = torch.zeros(bs, self.initPad, T).cuda(device)
        xaL = [torch.cat( (xaInit, tmpPad), 1 )]
        xbL = [torch.cat( (xbInit, tmpPad), 1 )]
        for layer in range(self.layerNum):
            xatmp, xbtmp = eval('iRevNetUtils.iRevNetBlock_forward('+\
                        'xaL['+str(layer)+'], xbL['+str(layer)+'],'+\
                        'self.Filt'+str(layer+1)+', self.decimate['+str(layer)+'])')
            xaL.append(xatmp)
            xbL.append(xbtmp)
            
        phi = torch.cat((xaL[-1], xbL[-1]),1)
        mask = self.maskEst(phi)
        phi2 = mask*phi
        
        ytmp = torch.split(phi2, self.chNum[-1], dim=1) 
        yaL = [ytmp[0]]
        ybL = [ytmp[1]]
        
        for layer in range(self.layerNum):
            yatmp, ybtmp = eval('iRevNetUtils.iRevNetBlock_backward('+\
                        'yaL['+str(layer)+'], ybL['+str(layer)+'],'+\
                        'self.Filt'+str(self.layerNum-layer)+', self.decimate['+str(self.layerNum-layer-1)+'])')
            yaL.append(yatmp)
            ybL.append(ybtmp)
        y = iRevNetUtils.initSplitInv(yaL[-1][:,0,:].unsqueeze(1), ybL[-1][:,0,:].unsqueeze(1))
        return y, phi, mask
        
    