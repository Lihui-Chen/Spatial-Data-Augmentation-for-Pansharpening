# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
          Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).

 Interface:
           ERGAS_index = ERGAS(I1,I2,ratio)

 Inputs:
           I1:             First multispectral image;
           I2:             Second multispectral image;
           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
 
 Outputs:
           ERGAS_index:    ERGAS index.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.    
"""

import numpy as np
import math

def ERGAS(I1,I2,ratio):

    I1 = I1.astype('float64')
    I2 = I2.astype('float64')

    Err = I1-I2

    ERGAS_index=0
    
    for iLR in range(I1.shape[2]):
        ERGAS_index = ERGAS_index + np.mean(Err[:,:,iLR]**2, axis=(0, 1))/(np.mean(I1[:,:,iLR], axis=(0, 1)))**2    
    
    ERGAS_index = (100/ratio) * math.sqrt((1/I1.shape[2]) * ERGAS_index)       
            
    return np.squeeze(ERGAS_index)

import numpy as np
import torch

EPS=1e-7

def ERGAS(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mean_real = mean_real**2
        mse = np.mean((img_fake_ - img_real_)**2)
        mean_real = np.maximum(mean_real, EPS)
        return 100.0 / scale * np.sqrt(mse / mean_real)
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        means_real = means_real**2
        means_real = np.maximum(means_real, torch.ones_like(mean_real)*EPS)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100.0 / scale * np.sqrt((mses/means_real).mean())
    else:
        raise ValueError('Wrong input image dimensions.')
    
def ERGAS_GPU(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.double()
    img_real_ = img_real.dobule()
    if len(img_fake_.shape) == 2:
        mean_real = img_real_.mean()
        mean_real = mean_real**2
        mse = ((img_fake_ - img_real_)**2).mean()
        mean_real = torch.max(mean_real, EPS)
        return 100.0 / scale * torch.sqrt(mse / mean_real)
    elif len(img_fake_.shape) == 3:
        means_real = img_real_.reshape(img_real_.shape[0], -1).mean(axis=1)
        means_real = means_real**2
        means_real = torch.max(means_real, torch.ones_like(means_real)*EPS)
        mses = ((img_fake_ - img_real_)**2).reshape(img_fake_.shape[0], -1).mean(axis=0)
        return 100.0 / scale * torch.sqrt((mses/means_real).mean())
    else:
        raise ValueError('Wrong input image dimensions.')