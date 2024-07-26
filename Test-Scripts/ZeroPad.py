import cmath
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zeroPad2DMatrix(layer_wt,n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = layer_wt.size()[3]
    p = (k-1)/2
    q = (k+1)/2
    I = torch.eye(n).to(device);
    ind1 = torch.arange(0,p)
    ind2 = torch.arange(p,k)
    ind3 = torch.arange(k,n)
    indices = torch.cat((ind2,ind3,ind1))
    indices=indices.type(torch.int64)
    perm = I[indices].to(device)
    perm_mat = perm.unsqueeze(0).unsqueeze(0)
    pad_left = 0
    pad_right = n - k
    pad_top = 0
    pad_bottom = n - k
    # Apply padding
    padded_wt = torch.nn.functional.pad(layer_wt, (pad_left, pad_right, pad_top, pad_bottom)).to(device)
    perm_mat_tr = torch.transpose(perm_mat,2,3)
    padded_final = torch.matmul(perm_mat,torch.matmul(padded_wt,perm_mat_tr))
    return padded_final

def deZeroPad2DMatrix(layer_wt,k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = layer_wt.size()[-1]
    p = (k-1)/2
    q = (k+1)/2
    I = torch.eye(n).to(device);
    ind1 = torch.arange(0,p)
    ind2 = torch.arange(p,k)
    ind3 = torch.arange(k,n)
    indices = torch.cat((ind2,ind3,ind1))
    indices=indices.type(torch.int64)
    perm = I[indices].to(device)
    #perm_mat = perm.unsqueeze(0).unsqueeze(0)

    # Apply padding
    perm_mat_tr = torch.transpose(perm,-2,-1)

    unpadded_wt = torch.matmul(perm_mat_tr,torch.matmul(layer_wt,perm))

    layer = unpadded_wt[:,:,:k,:k].to(device)
    return layer