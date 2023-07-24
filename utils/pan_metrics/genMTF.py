# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Generate a bank of filters shaped on the MTF of the sensor. Each filter
           corresponds to a band acquired by the sensor. 
 
 Interface:
           h = genMTF(ratio, sensor, nbands)

 Inputs:
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
           nbands:             Number of spectral bands.

 Outputs:
           h:                  Gaussian filter mimicking the MTF of the MS sensor.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.         
"""

import numpy as np
import torch
from .tools import fir_filter_wind, gaussian2d, kaiser2d, fir_filter_wind_torch, gaussian2d_torch, kaiser2d_torch
import einops

def  genMTF(ratio, sensor, nbands, anisotropic=False):
    eps = 1e-6
    N = 41
    if 'QB' in sensor:
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif 'IK' in sensor :
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif 'GE1' in sensor or 'WV4' in sensor:
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
    elif 'WV2' in sensor:
        GNyq = np.array([0.35]*(nbands-1)+[0.27], dtype='float32')
    elif 'WV3' in sensor:
        GNyq = np.asarray([0.325,0.355,0.360,0.350,0.365,0.360,0.335,0.315],dtype='float32') 
    elif 'P1B' in sensor:
        GNyq = np.asarray([0.28,0.28,0.29,0.28],dtype='float32') 
    elif 'SP7' in sensor:
        GNyq = np.asarray([0.33,0.33,0.33,0.33],dtype='float32')
    elif sensor.lower() =='random':
        # GNyq = np.random.normal(loc=0.3, scale=0.03, size=nbands)
        GNyq = np.random.uniform(low=0.2, high=0.4, size=nbands)
        if anisotropic:
            GNyq2 = np.random.uniform(low=0.2, high=0.4, size=nbands)
    elif sensor.lower() =='random_pan':
        # GNyq = np.random.normal(loc=0.3, scale=0.03, size=nbands)
        GNyq = np.random.uniform(low=0.05, high=0.25, size=nbands)
    else:
        GNyq = 0.3 * np.ones(nbands)   
    """MTF"""
    fcut = 1/ratio
    h = np.zeros((N,N,nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[ii])))
        if anisotropic:
            alpha2 = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq2[ii])))
            alpha = (alpha, alpha2)
        H=gaussian2d(N, alpha, anisotropic=anisotropic)
        Hd=H/np.max(H)
        w=kaiser2d(N,0.5)
        h[:,:,ii] = np.real(fir_filter_wind(Hd,w))
        
    return h


# -*- encoding: utf-8 -*-
'''
Copyright (c) 2020 Lihui Chen
All rights reserved. This work should only be used for nonprofit purposes.

@file        : genMTF.py
@Date        : 2022/04/05
@Author      : Lihui Chen
@version     : 1.0
@description : 
@reference   :
'''
def genMTF_torch(ratio, sensor, nbands=None, device=torch.device('cpu'), anisotropic=False):
    with torch.no_grad():
        tensor = torch.cuda.FloatTensor if device.type=='cuda' else torch.FloatTensor
        eps = 1e-6
        N = 41
        if 'QB' in sensor:
            GNyq = torch.tensor([0.34, 0.32, 0.30, 0.22], device=device)    # Band Order: B,G,R,NIR
        elif 'IK' in sensor:
            GNyq = torch.tensor([0.26,0.28,0.29,0.28], device=device)    # Band Order: B,G,R,NIR
        elif 'GE1' in sensor or 'WV4' in sensor:
            GNyq = torch.tensor([0.23,0.23,0.23,0.23], device=device)    # Band Order: B,G,R,NIR
        elif 'WV2' in sensor:
            GNyq = torch.tensor([0.35]*nbands+[0.27], device=device)
        elif 'WV3' in sensor:
            GNyq = torch.tensor([0.325,0.355,0.360,0.350,0.365,0.360,0.335,0.315], device=device) 
        elif 'P1B' in sensor:
            GNyq = torch.tensor([0.28,0.28,0.29,0.28], device=device) 
        elif 'SP7' in sensor:
            GNyq = torch.tensor([0.33,0.33,0.33,0.33], device=device)
        elif sensor.lower() =='random':
            GNyq = torch.nn.init.uniform_(tensor(nbands), 0.2, 0.4)
            if anisotropic:
                GNyq2 = torch.nn.init.uniform_(tensor(nbands), 0.2, 0.4)
        else:
            GNyq = 0.3 * torch.ones_like(tensor(nbands))
            
        """MTF"""
        fcut = 1/ratio
        h = torch.zeros_like(tensor(N,N,nbands))
        alpha = torch.sqrt(((N-1)*(fcut/2))**2/(-2*torch.log(GNyq)))
        if anisotropic:
            alpha2 = torch.sqrt(((N-1)*(fcut/2))**2/(-2*torch.log(GNyq2)))
            alpha = (alpha, alpha2)
        H=gaussian2d_torch(N, alpha, anisotropic=anisotropic)
        Hd=H/einops.reduce(H, 'h w c -> c', 'max')
        w=kaiser2d_torch(N, 0.5, device=device)
        h = torch.real(fir_filter_wind_torch(Hd,w))
        # h = torch.real(h)
    return h