# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:58:16 2020

@author: acoust
"""
import os, sys, glob
import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('../')
from modules import takeModules as tm
from iRevNet import modelDifinition


##################################################################################
# flag
deviceNum  = 0

##################################################################################
# exp. param
layerNum = 6


#filt = 'UNet5SpecNorm'
filt = 'LinearNoBiasUNet5SpecNorm'

red = 4


maskEstimator = 'binary'
#maskEstimator = 'UNet5Sigmoid'

lossMode = 'SDR'




# training data directory
cleanDir  = 'D:/sound_data/Voicebank_DEMAND/clean_testset_wav2'
noisyDir  = 'D:/sound_data/Voicebank_DEMAND/noisy_testset_wav2'

# save dnn directory
dnn_dir  = './dnn_dir/' 
if(os.path.isdir(dnn_dir)==False):
    os.mkdir(dnn_dir)
    
# train parameter
speechPerSet = 2048
batchSize = 16
Log_reg = 10**(-6)
valRatio = 0.1
speechLen = 2**15

maxEpoch = 500


initPad=red-1
##################################################################################
saveName = \
'iRevNet_L'+str(layerNum)+\
'R'+str(initPad+1)+\
'_'+filt+\
'_'+maskEstimator+\
'_'+lossMode+\
'_bs'+str(batchSize)+\
'_bpl'+str(speechLen)+\
'_vr'+str(valRatio)\
+'_ep'+str(maxEpoch)
fileName = dnn_dir+saveName

testDir = 'D:/sound_data/test_iRevNet_pytorch'
if(os.path.isdir(testDir)==False):
            os.mkdir(testDir)
#print(saveName)

condDir = testDir+'/'+saveName
if(os.path.isdir(condDir)==False):
    os.mkdir(condDir)
            

##################################################################################


estClean = modelDifinition.iRevNetMasking( layerNum, filt, initPad, maskEstimator).cuda(deviceNum)
estClean.load_state_dict(torch.load(fileName))


sdataFns  = glob.glob(cleanDir + "/*.wav")
xdataFns  = glob.glob(noisyDir + "/*.wav")
testNum = len(sdataFns)

for utter in range(testNum):
    sys.stdout.write('\rTestSet: '+str(utter+1)+'/'+str(testNum)) 
    sys.stdout.flush()
    s = torch.from_numpy(tm.wavread(sdataFns[utter])[0]).cuda(deviceNum)
    x = torch.from_numpy(tm.wavread(xdataFns[utter])[0]).cuda(deviceNum)
    sLen = len(s) 
    zp = speechLen - sLen%speechLen
    s = torch.cat( (s, torch.zeros(zp).cuda(deviceNum)), 0 ).unsqueeze(0)
    x = torch.cat( (x, torch.zeros(zp).cuda(deviceNum)), 0 ).unsqueeze(0)    
    y, phi, mask = estClean(x)
    y = y.detach()

    s = s[0][:sLen]
    x = x[0][:sLen]
    y = y[0][:sLen]
    
    saveFn = condDir+'/'+sdataFns[utter][len(cleanDir)+1:]
    tm.wavwrite(saveFn, y.cpu().numpy(), 16000)
    
sys.stdout.write('\n')
