import os, time, sys
import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules import takeModules as tm
from modules import myLossFunction as myLF
from iRevNet import modelDifinition


##################################################################################
# flag
testFlag = 0 # 0:full train mode 1:test mode (few files) 
deviceNum  = 0

##################################################################################
# exp. param
layerNum = 6

filt = 'UNet5SpecNorm'
#filt = 'LinearNoBiasUNet5SpecNorm'

red = 4

maskEstimator = 'binary'
#maskEstimator = 'UNet5Sigmoid'
#maskEstimator = 'insNormUNet5Sigmoid'

lossMode = 'SDR'

# training data directory
cleanDir  = 'D:/sound_data/Voicebank_DEMAND/clean_trainset_wav'
noisyDir  = 'D:/sound_data/Voicebank_DEMAND/noisy_trainset_wav'

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


maxEpoch         = 500
lr_init           = 0.0001

##################################################################################
initPad= red-1

saveName = \
'iRevNet_L'+str(layerNum)+\
'R'+str(initPad+1)+\
'_'+filt+\
'_'+maskEstimator+\
'_'+lossMode+\
'_bs'+str(batchSize)+\
'_bpl'+str(speechLen)+\
'_vr'+str(valRatio)
fileName = dnn_dir+saveName
print(fileName)

##################################################################################
trainData, validData = tm.dataLoad(clean_dir = cleanDir, noisy_dir = noisyDir,
                                 val_ratio = valRatio, speech_per_set = speechPerSet,
                                 test_flag = testFlag)

estClean = modelDifinition.iRevNetMasking( layerNum, filt, initPad, maskEstimator).cuda(deviceNum)
optimizer = optim.Adam(estClean.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

lossFunc = eval('myLF.'+lossMode)


for param in estClean.parameters():
    nn.init.normal_(param, 0.0, 0.001)
    


print("train start") 
start = time.time()
olddatetime = 'None' 
trainLoss = np.array([])
validLoss = np.array([])
for epoch in range(1, maxEpoch+1):
#    print(saveName)
    sumLoss  = 0.0
    sumSDR = 0.0
    perm1 = np.random.permutation( len(trainData) )
    start = time.time()
    
    for setNum in range( len(trainData) ):
        sys.stdout.write('\repoch: '+str(epoch)+' TrnSet: '+str(setNum+1)+'/'+str(len(trainData))) 
        sys.stdout.flush()
        trainMiniSet = tm.list_to_gpu( trainData[ perm1[setNum] ], deviceNum )
        perm2 = np.random.permutation( len(trainMiniSet) )
        batchNum = len(trainMiniSet)//batchSize
        for utter in range(batchNum):
            optimizer.zero_grad()
            s = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
            x = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
            for bs in range( batchSize ):
                stmp = trainMiniSet[perm2[bs+utter*batchSize]][0]
                xtmp = trainMiniSet[perm2[bs+utter*batchSize]][1]
                
                if len(stmp)>speechLen:
                    st = np.random.randint(len(stmp)-speechLen)
                    end = st+speechLen
                    stmp = stmp[st:end]
                    xtmp = xtmp[st:end]
                else:
                    zLen = speechLen - len(stmp)
                    tmpPad = torch.zeros((zLen)).cuda(deviceNum)
                    stmp = torch.cat( (stmp, tmpPad), 0 )
                    xtmp = torch.cat( (xtmp, tmpPad), 0 )
                
                stmp =  stmp.unsqueeze(0)
                xtmp =  xtmp.unsqueeze(0)
                s = torch.cat( (s,stmp), 0)
                x = torch.cat( (x,xtmp), 0)
                
            y, _, _ = estClean(x)
            loss = lossFunc(s, x, y)
            loss.backward()
            optimizer.step()
            
            sumLoss += loss.detach().cpu().numpy()
                
    sys.stdout.write('\n')
    
    print("time/epoch(Train):"+str(time.time() - start))
    print("avg. loss:"+str(sumLoss/batchNum))
    trainLoss= np.append(trainLoss, sumLoss/batchNum)
    
    if valRatio !=0:
        start = time.time()
        for param in estClean.parameters():
            param.requires_grad = False
        sumLoss_val  = 0.0
        sumSDR_val = 0.0
        perm1_val = np.random.permutation( len(validData) )
        for setNum in range( len(validData) ):
            sys.stdout.write('\repoch: '+str(epoch)+' ValSet: '+str(setNum+1)+'/'+str(len(validData))) 
            sys.stdout.flush()
            validMiniSet = tm.list_to_gpu( validData[ perm1_val[setNum] ], deviceNum )
            perm2_val = np.random.permutation( len(validMiniSet) )
            batchNum_val = len(validMiniSet)//batchSize
            for utter in range(batchNum_val):
                optimizer.zero_grad()
                s_val = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
                x_val = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
                for bs in range( batchSize ):
                    stmp = validMiniSet[perm2_val[bs+utter*batchSize]][0]
                    xtmp = validMiniSet[perm2_val[bs+utter*batchSize]][1]
                    
                    if len(stmp)>speechLen:
                        st = np.random.randint(len(stmp)-speechLen)
                        end = st+speechLen
                        stmp = stmp[st:end]
                        xtmp = xtmp[st:end]
                    else:
                        zLen = speechLen - len(stmp)
                        tmpPad = torch.zeros((zLen)).cuda(deviceNum)
                        stmp = torch.cat( (stmp, tmpPad), 0 )
                        xtmp = torch.cat( (xtmp, tmpPad), 0 )
                    
                    stmp =  stmp.unsqueeze(0)
                    xtmp =  xtmp.unsqueeze(0)
                    s_val = torch.cat( (s_val,stmp), 0)
                    x_val = torch.cat( (x_val,xtmp), 0)
                    
                s_val.detach()
                x_val.detach()
                y_val, _, _ = estClean(x_val)
                d_val = x_val-y_val
                n_val = x_val-s_val
                loss = lossFunc(s_val, x_val, y_val)
                sumLoss_val += loss.detach().cpu().numpy()
                    
        sys.stdout.write('\n')
        
        print("time/epoch(Valid):"+str(time.time() - start))
        print("avg. loss:"+str(sumLoss_val/batchNum_val))
        validLoss= np.append(validLoss, sumLoss_val/batchNum_val)
        for param in estClean.parameters():
            param.requires_grad = True
        
if testFlag == 0: 
    print('save DNN at epoch '+str(epoch))
    torch.save(estClean.state_dict(), fileName+'_ep'+str(epoch))
else:
    print('test mode (do not save)')