# -*- encoding: utf-8 -*-
'''
Copyright (c) 2020 Lihui Chen
All rights reserved. This work should only be used for nonprofit purposes.

@file        : scale_change.py
@Date        : 2022/04/05
@Author      : Lihui Chen
@version     : 1.0
@description : 

@reference:

'''

import numpy as np
import torch
import torch.nn.functional as F
from utils.pan_metrics.interp23 import interp23
from utils.pan_metrics.imresize import imresize
from data.trans_data import multidata

@multidata
def scale_change(img, scale, method='bicubic', op_type='np'):
    if op_type=='np':
        if method=='interp23':
            img = interp23(img, scale)
        else:
            imgdict = imresize(img, scale, method=method)
    elif op_type=='torch':
        img = F.interpolate(img, scale_factor=scale, mode=method, align_corners=False) 
    else:
        raise TypeError('Cannot recognize this [%s] type of data for augmentation'%op_type)
    return img