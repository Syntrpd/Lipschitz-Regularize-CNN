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

#ADMM Frobenius Normalize
def LipConstrainLayer(layer,n):
    layer = layer.to(device)
    s = layer.shape
    k = s[3]

    # Constants (example values, replace these with actual data)
    H0 = zeroPad2DMatrix(layer,n).detach()
    H0 = torch.reshape(H0,(s[0]*s[1],n,n)).to(device)

    # Initialize optimization variables (H and lambda)
    H = torch.rand_like(H0,dtype = torch.float32).to(device)
    print(H.shape)
    U = torch.zeros((s[0]*s[1],n,n),dtype=torch.complex64).to(device)
    Hf = torch.zeros((s[0]*s[1],n,n),dtype=torch.complex64).to(device)

    pri_res = 100
    rho = 0.01

    #i = 0
    #optimizer = torch.optim.Adam([H], lr=0.01)
    #while pri_res>0.5:
    for i in range(1000):
        # Optimize H
        x = torch.fft.ifft2(Hf-U)
        H = (2*H0 + (rho) * torch.real(x)*n*n)/(2+rho*n*n)
        H = torch.reshape(H,(s[0],s[1],n,n))
        H = deZeroPad2DMatrix(H,k)
        H = zeroPad2DMatrix(H,n)
        H = torch.reshape(H,(s[0]*s[1],n,n))

        H_fourier = torch.fft.fft2(H)

        #Optimize Hf

        dual_res = Hf
        Hf = H_fourier + U
        H_frob = torch.clamp(torch.sqrt(torch.sum(torch.square(torch.abs(Hf)),dim = 0)),min=1)
        #H_frob = torch.sum(torch.abs(Hf),dim = 0)
        #print(H_frob)
        s_f = H_frob.shape
        Hf = torch.div(Hf,torch.reshape(H_frob,(1,n,n)))
        dual_res = torch.norm(torch.fft.ifft2(Hf - dual_res), p='fro')*rho
        

        # Update U
        U = U + H_fourier - Hf
        pri_res = torch.norm(H_fourier - Hf, p='fro')
        #i=i+1
        if(i%100 == 0):
            print(f'Step {i}: Residual:{pri_res+dual_res}')
        
        del x,H_fourier,s_f
    # Results
    print(f'Optimized H:\n{H}')