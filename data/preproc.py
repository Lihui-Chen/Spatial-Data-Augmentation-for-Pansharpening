import random
import numpy as np
import torch.nn as nn
from data import trans_data


def get_patch(imgdict:dict, scale_dict:dict, patch_size:int):
    '''
    imgdict: a list of images whose resolution increase with index of the list
    scale_dict: list of scales for the corresponding images in imglist
    patch_size: the patch size for the fisrt image to be cropped.
    '''
    # minkey = min(scale_dict.key(), key=lambda x:scale_dict[x])
    # iw, ih = imgdict[minkey].shape[-2]
    ih, iw = min(imgdict.values(), key= lambda x:x.shape[1]).shape[:2]
    min_scale = min(scale_dict.values())
    # if min_scale!=1: scale_dict={key:value/min_scale for key, value in scale_dict.items()}
    ix = random.randrange(0, int(ih/min_scale) - patch_size + 1)
    iy = random.randrange(0, int(iw/min_scale) - patch_size + 1)
    sizedict = {t_key: (ix*t_scale, iy*t_scale, patch_size*t_scale)
                for t_key, t_scale in scale_dict.items()}
    out_patch = {t_key: imgdict[t_key][ix:ix+t_psize, iy:iy+t_psize,:]
                 for t_key, (ix, iy, t_psize) in sizedict.items()}
    return out_patch


def augment(input, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    @trans_data.multidata
    def _augment(img, hflip, vflip, rot90):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img
    return _augment(input, hflip, vflip, rot90)


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img

def degradation(img, degraded_type: tuple):
    for de_type in degraded_type:
        if de_type == 'blur':
            img = blur(img)
        elif de_type == 'downsample':
            img = downsample(img)
        elif de_type == 'noising':
            img = add_noise(img)
    return img

def blur(img, kernel):
    pass

def downsample(img, down_type):
    pass


def add_noise(x, noise='.'):
    if noise != '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)
        x_noise = x + noise
        # x_noise = x.astype(np.int16) + noises.astype(np.int16)
        # x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x