import cmath
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def createFourierMatrix(k,n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i = cmath.sqrt(-1)
    val = cmath.exp(-2*cmath.pi*i/n)
    p = (k-1)/2
    q = (k+1)/2
    F = torch.zeros(n*n,k*k).to(device)
    F = F.type(torch.complex64)

    f = torch.zeros(n,1).to(device)
    f = f.type(torch.complex64)
    f_u = torch.zeros(n*n,1).to(device)
    f_u = f_u.type(torch.complex64)
    for u in range(n):
        f_u[u*n:(u+1)*n]=val**u
        f[u]=val**u;

    f_v = f.repeat(n,1);
    for u in range(k):
        for v in range(k):
            a=0
            b=0
            if(u<=p):
                a = n-p+u;
            else:
                a = u-p;


            if(v<=p):
                b = n-p+v;
            else:
                b = v-p;

            F[:,(u*k+v)]=((torch.pow(f_u,(a)))*(torch.pow(f_v,(b)))).flatten();

    return F

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



def computeLayerLipschitzFourier(layer_wt,n):
    layer_wt_padded = zeroPad2DMatrix(layer_wt,n)
    layer_pf=torch.fft.fft2(layer_wt_padded)
    layer_fperm = torch.permute(layer_pf,(2,3,0,1))
    sing = torch.linalg.svdvals(layer_fperm)
    lip = torch.max(torch.abs(sing))
    return lip