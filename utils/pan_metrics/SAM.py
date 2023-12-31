# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Spectral Angle Mapper (SAM).
 
 Interface:
           [SAM_index,SAM_map] = SAM(I1,I2)

 Inputs:
           I1:         First multispectral image;
           I2:         Second multispectral image.
 
 Outputs:
           SAM_index:  SAM index;
           SAM_map:    Image of SAM values.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""

import numpy as np
import math
import torch

def  SAM(I1,I2):
    
    M = I2.shape[0]
    N = I2.shape[1]
    
    prod_scal = np.zeros((M,N))
    norm_orig = np.zeros((M,N))
    norm_fusa = np.zeros((M,N))
    for iM in range(M):
        for iN in range(N):            
            h1 = I1[iM,iN,:]
            h2 = I2[iM,iN,:]
            prod_scal[iM,iN] = h1.flatten() @ h2.flatten()
            norm_orig[iM,iN] = h1.flatten() @ h1.flatten()
            norm_fusa[iM,iN] = h2.flatten() @ h2.flatten()
    
    
    prod_norm = np.sqrt(norm_orig * norm_fusa)
    prod_map = prod_norm
    prod_map[prod_map == 0] = 2 * 10**(-16)
    
    SAM_map = np.arccos(prod_scal/prod_map)
    
    prod_scal = np.reshape(prod_scal, (M*N,1))
    prod_norm = np.reshape(prod_norm, (M*N,1))
    
    z = np.nonzero(prod_norm == 0)
    
    prod_scal[z]=[]
    prod_norm[z]=[]
    
    angolo = np.sum(np.arccos(prod_scal/prod_norm))/(prod_norm.shape[0])
    
    SAM_index = angolo*180/math.pi
    
    return SAM_index, SAM_map

def SAM_pytorch(im_true, im_fake):
    B, C, H, W = im_true.shape
    # C = im_true.size()[1]
    # H = im_true.size()[2]
    # W = im_true.size()[3]
    esp = 1e-10
    # Itrue = im_true.clone()#.resize_(C, H*W)
    # Ifake = im_fake.clone()#.resize_(C, H*W)
    # print('first', im_fake.mean())
    nom = torch.mul(im_true, im_fake).sum(dim=1)#.resize_(H*W)
    im_true = torch.sqrt((im_true*im_true).sum(dim=1).clamp(min=esp))
    im_fake = torch.sqrt((im_fake*im_fake).sum(dim=1).clamp(min=esp))

    # denominator = im_true.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
    #               im_fake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
    denominator = (im_true*im_fake).clamp(min=esp)
    # print('second',im_fake.mean())
    sam = torch.div(nom, denominator)#.acos()
    sam = 1-sam
    # sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W* B) #/ np.pi * 180

    return sam_sum