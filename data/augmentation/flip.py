import numpy as np
import torch
import kornia
from data.trans_data import multidata

def randomflip(img, op_type='np'):
    vflip = np.random.random()<0.5
    hflip = np.random.random()<0.5
    if op_type=='np':
        if vflip: img=np_vflip(img)
        if hflip: img =np_hflip(img)
    elif op_type=='torch':
        if vflip: img=torch_vflip(img)
        if hflip: img=torch_hflip(img)
    else:
        raise TypeError('Cannot recognize this [%s] type of data for augmentation'%op_type)
    return img

@multidata
def np_vflip(img):
    return img[ ::-1, :, :]

@multidata
def np_hflip(img):
    return img[:, ::-1, :]



@multidata
def torch_vflip(img):
    return img[:, :, ::-1, :]

@multidata
def torch_hflip(img):
    return img[:, :, :, ::-1]