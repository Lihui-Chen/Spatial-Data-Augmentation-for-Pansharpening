import numpy as np
import torch
import kornia
from data.trans_data import multidata

def randomrot(img, op_type='np'):
    deg = np.random.randint(0, 4)
    if op_type=='np':
        img = np_rot(img, deg)
    elif op_type=='torch':
        img = torch_rot(img, deg)
    else:
        raise TypeError('Cannot recognize this [%s] type of data for augmentation'%op_type)
    return img

@multidata
def np_rot(img, deg):
    img = np.rot90(img, k=deg, axes=[0,1])
    return img

@multidata
def torch_rot(img, deg):
    img = torch.rot(img, k=deg, dims=[2,3])
    return img
