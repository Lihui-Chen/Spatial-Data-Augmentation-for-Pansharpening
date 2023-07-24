# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )

Adapted versions of the functions used in:
    Python Code on GitHub: https://github.com/sergiovitale/pansharpening-cnn-python-version    
    Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').    
"""

import numpy as np
from zmq import device
from data.kernel import anisotropic_gaussian_kernel, cal_sigma
import torchvision.transforms.functional as tvF


def fir_filter_wind(Hd,w):
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    #h=h/np.sum(h)
    
    return h

def gaussian2d (N, std, anisotropic=False):
    """author: Lihui Chen,
       Modified from Gemine Vivone's toolbox.
    Args:
        N (_type_): the support of MTF
        std (_type_): standard deviation for MTF
        anisotropic (bool, optional): if use the anisotropic kernel or not. Defaults to False.

    Returns:
        _type_: _description_
    """    
    t=np.arange(-(N-1)/2,(N+1)/2)
        #t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    if anisotropic:
        sigma = cal_sigma(std[0], std[1], radians=np.random.uniform(0, np.pi))
        w = anisotropic_gaussian_kernel(N, sigma)
        # w= np.exp(-0.5*(t1/std[0].astype(np.double))**2)*np.exp(-0.5*(t2/std[1].astype(np.double()))**2)  
    else:
        std=np.double(std)
        w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2)
    return w
    
def kaiser2d (N, beta):
    t=np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    #t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w

# -*- encoding: utf-8 -*-
'''
Copyright (c) 2022 Lihui Chen
All rights reserved. This work should only be used for nonprofit purposes.

@file        : tools.py
@Date        : 2022/04/05
@Author      : Lihui Chen
@version     : 1.0
@description : torch version of generate MTF for pansharpenig
@reference   :
'''

import torch
from data.augmentation.util import Interp1d

def fir_filter_wind_torch(Hd,w):
    """batch fir_filter generation with size (kh, kw, batch)

    Args:
        Hd (torch.Tensor): size of (kh, kw, batch)
        w (kaiser_winder): size of (kh, kw)

    Returns:
        h: spatial filter with kaiser_window
    """
    hd=torch.rot90(torch.fft.fftshift(torch.rot90(Hd,2, dims=(0, 1))),2, dims=(0,1))
    h=torch.fft.fftshift(torch.fft.ifft2(hd, dim=(0, 1)))
    h=torch.rot90(h,2, dims=(0,1))
    h=h*w.unsqueeze(dim=-1)
    #h=h/np.sum(h)
    return h

def gaussian2d_torch(N, std, anisotropic=False):
    device = std[0].device if isinstance(std, (list, tuple)) else std.device
    t=torch.arange(-(N-1)/2,(N+1)/2, device=device)
    #t=np.arange(-(N-1)/2,(N+2)/2)
    # t1,t2=torch.meshgrid(t,t) 
    t1, t2 = torch.meshgrid(t,t, indexing='xy') 
    
    if not anisotropic:
        t1 = t1.unsqueeze(dim=-1)
        t2 = t2.unsqueeze(dim=-1)
        std = std.unsqueeze(dim=0).unsqueeze(dim=0).expand(*t1.shape[:2], -1)
        w = torch.exp(-0.5*(t1/std)**2)*torch.exp(-0.5*(t2/std)**2) 
    else:
        tensor = torch.cuda.FloatTensor if device.type=='cuda' else torch.FloatTensor
        theta = torch.nn.init.uniform_(tensor(std[0].shape), 0, torch.pi)
        # theta = torch.zeros(std[0].shape, device=device)
        theta = rotate_matrix(theta)
        v =  theta@ torch.tensor([1.0, 0], device=device).unsqueeze(dim=-1)
        v = v.transpose(1,2).repeat((1,2,1))
        v[:,1,0], v[:, 1, 1] = v[:, 1, 1], -v[:, 1, 0]
        d1 = (1/(std[0]*std[0])).unsqueeze(dim=-1).unsqueeze(dim=-1)
        d1 = torch.cat([d1, torch.zeros_like(d1)], dim=2)
        d2 = (1/(std[1]*std[1])).unsqueeze(dim=-1).unsqueeze(dim=-1)
        d2 = torch.cat([torch.zeros_like(d2), d2], dim=2)
        D = torch.cat((d1, d2), dim=1)
        # V = torch.tensor([[v[0,:], v[1,:]], [v[0,:], -v[1,:]]], device=device)
        # D = torch.permute(2, 0, 1)
        Sigma = v@D@torch.linalg.inv(v)
        t1 = t1.view(N, N, 1, 1)
        t2 = t2.view(N, N, 1, 1)

        t = torch.cat((t1, t2), dim=3).unsqueeze(dim=0).repeat((std[0].shape[0], 1, 1, 1, 1))
        Sigma = Sigma.view(std[0].shape[0], 1, 1, 2, 2).repeat((1, N, N, 1, 1))
        Sigma = (-0.5*t@Sigma@t.transpose(-1, -2)).squeeze()
        w = torch.exp(Sigma).permute((1,2,0))
        # std = [tmp.unsqueeze(dim=0).unsqueeze(dim=0).expand(*t1.shape[:2], -1) for tmp in std] 
        # metrics = (-0.5*((t1/std[0])**2+(t2/std[1])**2))
        # w = tvF.rotate(w, np.random.uniform(0, 90), expand=True)
        # w= torch.exp()
        # w = tvF.rotate(w, np.random.uniform(0, 90), expand=True)
    return w

def get_sigma(N, theta, std):
    device = std[0].device if isinstance(std, (list, tuple)) else std.device
    tensor = torch.cuda.FloatTensor if device.type=='cuda' else torch.FloatTensor
    theta = torch.nn.init.uniform_(tensor(std[0].shape), 0, 180)
    theta = rotate_matrix(theta)
    v =  theta@ torch.tensor([1.0, 0], device=device).unsqueeze(dim=-1)
    v = v.transpose(1,2).repeat((1,2,1))
    v[:,1,0], v[:, 1, 1] = v[:, 1, 1], -v[:, 1, 0]
    d1 = (std[0]*std[0]).unsqueeze(dim=-1).unsqueeze(dim=-1)
    d1 = torch.cat([d1, torch.zeros_like(d1)], dim=2)
    d2 = -(std[1]*std[1]).unsqueeze(dim=-1).unsqueeze(dim=-1)
    d2 = torch.cat([torch.zeros_like(d2), d2], dim=2)
    D = torch.cat((d1, d2), dim=1)
    # V = torch.tensor([[v[0,:], v[1,:]], [v[0,:], -v[1,:]]], device=device)
    # D = torch.permute(2, 0, 1)
    Sigma = v@D@torch.linalg.inv(v)
    t1 = t1.view(N, N, 1, 1)
    t2 = t2.view(N, N, 1, 1)
    t = torch.cat((t1, t2), dim=3).unsqueeze(dim=0).repeat((std[0].shape[0], 1, 1, 1, 1))
    Sigma = Sigma.view(std[0].shape[0], 1, 1, 2, 2).repeat((1, N, N, 1, 1))
    Sigma = (-0.5*t@Sigma@t.transpose(-1, -2)).squeeze()
    return Sigma

def rotate_matrix(theta):
    a11 = torch.cos(theta).view(-1, 1, 1)
    a12 = -torch.sin(theta).view(-1, 1, 1)
    a21 = torch.sin(theta).view(-1, 1, 1)
    a22 = torch.cos(theta).view(-1, 1, 1)

    a1 = torch.cat([a11, a12 ], dim=2)
    a2 = torch.cat([a21, a22], dim=2)
    theta = torch.cat((a1, a2), dim=1)
    return theta

def kaiser2d_torch(N, beta, device):
    t=torch.arange(-(N-1)/2,(N+1)/2, device=device)/(N-1)
    #t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1,t2=torch.meshgrid(t,t, indexing='xy') #torch.meshgrid(t,t, indexing='xy') 
    t12=torch.sqrt(t1*t1+t2*t2)
    w1=torch.kaiser_window(N, False, beta=beta, device=device)
    w=Interp1d()(t,w1, t12)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    return w



