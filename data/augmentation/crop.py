import numpy as np
import torch
import kornia
from data.trans_data import multidata


def random_crop(img, patchsize, op_type='np'):
    if op_type=='np':
        if isinstance(img, (tuple, list)):
            ih, iw = img[0].shape[:2] 
        else:    
            ih, iw = img.shape[:2] 
        ix = np.random.randint(0, ih - patchsize + 1)
        iy = np.random.randint(0, iw - patchsize + 1)
        img = np_crop(img, (ix, iy), patchsize)
    elif op_type=='torch':
        if isinstance(img, (tuple, list)):
            ih, iw = img[0].shape[-2:] 
        else:
            ih, iw = img.shape[-2:] 
        ix = np.random.randint(0, ih - patchsize + 1)
        iy = np.random.randint(0, iw - patchsize + 1)
        img = torch_crop(img, (ix, iy), patchsize)
    else:
        raise TypeError('Cannot recognize this [%s] type of data for augmentation'%op_type)
    return img

@multidata
def np_crop(img, pos, size):
    return img[pos[0]:pos[0]+size, pos[1]:pos[1]+size, :]

@multidata
def torch_crop(img, pos, size):
    return img[:, :, pos[0]:pos[0]+size, pos[1]:pos[1]+size]

def dict_random_crop(imgdict:dict, patch_size:int, op_type='np'):
    '''
    imgdict: a list of images whose resolution increase with index of the list
    scale_dict: list of scales for the corresponding images in imglist
    patch_size: the patch size for the fisrt image to be cropped.
    '''
    if op_type=='np':
        sz_dict = {key:value.shape[0] for key, value in imgdict.items()}
    else:
        sz_dict = {key:value.shape[-1] for key, value in imgdict.items()}
    minkey = min(sz_dict.keys(), key=lambda x:sz_dict.get(x))
    scale_dict = {key: value//sz_dict[minkey] for key, value in sz_dict.items()}
    ih, iw = imgdict[minkey].shape[:2] if op_type=='np' else imgdict[minkey].shape[-2:]
    ix = np.random.randint(0, ih - patch_size + 1)
    iy = np.random.randint(0, iw - patch_size + 1)
    pos_size_dict = {t_key: (ix*t_scale, iy*t_scale, patch_size*t_scale)
                for t_key, t_scale in scale_dict.items()}
    if op_type=='np': 
        out_patch = {t_key: imgdict[t_key][ix:ix+t_psize, iy:iy+t_psize,:]
                 for t_key, (ix, iy, t_psize) in pos_size_dict.items()}
    elif op_type=='torch':
        out_patch = {t_key: imgdict[t_key][:, :, ix:ix+t_psize, iy:iy+t_psize]
                 for t_key, (ix, iy, t_psize) in pos_size_dict.items()}    
    return out_patch


def scale_adap_crop(img, scale, op_type='np'):
    if op_type=='np':
        if img.ndim == 2: img = img[:,:,np.newaxis]
        h, w, c = img.shape
        del_h =  h-(h//scale)*scale
        if del_h!=0: img = img[:-del_h, :, :]
        del_w = w-(w//scale)*scale
        if del_w != 0: img = img[:,:-del_w, :]
    elif op_type=='torch':
        b, c, h, w = img.shape
        del_h =  h-(h//scale)*scale
        if del_h != 0: img = img[:,:,:-del_h, :]
        del_w = w-(w//scale)*scale
        if del_w !=0: img = img[:, :, :, :-del_w]
    return img