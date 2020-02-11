# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:41:25 2020

@author: acoust
"""

import sys, scipy, glob
import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################################
# transfer to GPU 
def list_to_gpu(cpuList, deviceNum):
    gpuList = []
    for ii in range(len(cpuList)):
        gpuList.append( torch.from_numpy( cpuList[ii] ).float().cuda(deviceNum) )
    return gpuList

#####################################################################
# read & write audio file
def wavread(fn):
    fs, data = wavfile.read(fn)
    data     = (data.astype(np.float32) / 2**(15))
    return data, fs

def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int32")
    wavfile.write(fn, fs, data)
    
#####################################################################
# load clean & noisy set
def dataLoad(**kwargs):
    if 'clean_dir' in kwargs and 'noisy_dir' in kwargs:
        return dataLoad_CN(**kwargs)
    
def dataLoad_CN(clean_dir, noisy_dir, val_ratio, speech_per_set, test_flag): 
    c_files = glob.glob(clean_dir + "/" + "*.wav")
    n_files = glob.glob(noisy_dir + "/" + "*.wav")
    Num_wav   = len( c_files )
    if test_flag == 1:
        Num_wav = round(Num_wav*0.02)
    TrnNum    = int(Num_wav* (1-val_ratio)) # 全学習データの 90% を学習に使う
    DevNum    = Num_wav-TrnNum    # 残りは varidation 用
    perm      = np.random.permutation( Num_wav )
    TrnIndex  = perm[0:TrnNum]       # Training set
    DevIndex  = perm[TrnNum:Num_wav] # Varidation set

    S_all = []
    S_set = []
    V_all = []
    V_set = []
    
    
    print( 'Loading... (Training set)' )
    cnt   = 0
    for ii in range( TrnNum ):
        if( ii%int(TrnNum/20) == 0 ):
            sys.stdout.write( '\r   '+str(ii+1)+'/'+str( TrnNum ) )
            sys.stdout.flush()
        c_fn = c_files[TrnIndex[ii]]
        n_fn = n_files[TrnIndex[ii]]
        s, org_fs = wavread( c_fn )
        x, org_fs = wavread( n_fn ) 
        S_set.append( np.vstack([s,x]) )
        cnt += 1
        if(cnt == speech_per_set):
            cnt = 0
            S_all.append( S_set )
            S_set = []
    sys.stdout.write('\n')
    if len(S_set)>0:
        S_all.append( S_set )
    
    cnt   = 0
    print( 'Loading... (Validation set)' )
    for ii in range( DevNum ):
        if( ii%int(DevNum/20) == 0 ):
            sys.stdout.write( '\r   '+str(ii+1)+'/'+str( DevNum ) )
            sys.stdout.flush()
        c_fn = c_files[DevIndex[ii]]
        n_fn = n_files[DevIndex[ii]]
        s, org_fs = wavread( c_fn )
        x, org_fs = wavread( n_fn ) 
        V_set.append( np.vstack([s,x]) )
        cnt += 1
        if(cnt == speech_per_set):
            cnt = 0
            V_all.append( V_set )
            V_set = []
    sys.stdout.write('\n')
    if len(S_set)>0:
        V_all.append( V_set )
    return S_all, V_all

    