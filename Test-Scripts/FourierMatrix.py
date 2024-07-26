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