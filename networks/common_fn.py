# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023 Lihui Chen
All rights reserved. This work should only be used for nonprofit purposes.

@file        : common_fn.py
@Date        : 2023/07/24
@Author      : Lihui Chen
@version     : 1.0
@description : 
@reference   :
'''
import torch
import numpy as np
import torch.nn.functional as F
from utils.pan_metrics import genMTF
import random
import kornia
import einops

EPS = 1e-7

def shuffle_channel(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C//groups, H, W)
    x = x.transpose(1, 2)
    x = x.contiguous()
    x = x.view(B, C, H, W)
    return x

def layer_norm(x, eps=1e-6):
    dim_len = len(x.shape)
    ori = x
    b = x.shape[0]
    x = x.view(b, -1)
    mean_x = x.mean(dim=-1).view(b, *([1]*(dim_len-1)))
    std_x = x.std(dim=-1, unbiased=False).view(b, *([1]*(dim_len-1)))
    std_x = torch.max(std_x, torch.ones_like(std_x)*EPS)
    return (ori-mean_x)/std_x, mean_x, std_x

def instance_norm(x): 
    "x: Tensor"
    B,C = x.shape[:2]
    ori = x
    x = x.view(B, C, -1)
    mean_x = x.mean(dim=-1).view(B,C, 1, 1)
    std_x = x.std(dim=-1).view(B, C, 1,1)
    std_x = torch.max(std_x, torch.ones_like(std_x)*EPS)
    return (ori-mean_x)/std_x, mean_x, std_x

def calc_mean_std(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def normalize(input):
    size = input.shape
    assert(len(size)==4)
    N, C =  size[:2]
    # input_tmp = input.view(N, C, -1)
    input_max = input.max()
    input_min = input.min()
    if input_max==input_min and input_max!=0:
        input_min=0
    elif input_max==input_min and input_max==0:
        input_max = 1
    # input_max = input_tmp.max(dim=2).values
    # input_min = input_tmp.min(dim=2).values
    # input_min = torch.where((input_min!=0)|(input_min!=input_max), input_min, torch.tensor(1.0, device=input.device))
    # input_min = torch.where(input_max!=input_min, input_min, torch.tensor(0.0, device=input.device))
    # input_max = input_max.view(N, C, 1, 1)
    # input_min = input_min.view(N, C, 1, 1)
    return (input-input_min)/(input_max-input_min), input_max, input_min

def denormalize(input, input_max, input_min):
    return input*(input_max-input_min) + input_min

def data_norm(ms, pan):
    norm_min = min(ms.min(), pan.min())
    norm_max = max(ms.max(), pan.max())
    ms = (ms-norm_min)/(norm_max-norm_min)
    pan = (pan-norm_min)/(norm_max-norm_min)
    return ms, pan, norm_min, norm_max

def get_filter_kernel(type, direction=None):
    if type=='Sobel':
        if direction=='y':
            kernel = torch.tensor([[1.0, 2, 1], [0, 0 ,0], [-1, -2, -1]]).div_(8)
        else:
            kernel = torch.tensor([[1.0, 0, -1], [2, 0 ,-2], [1, 0, -1]]).div_(8)

    return kernel


def MTF_Down(hrms, scale, sensor='random', mtf:np.ndarray=None, anisotropic=False):
    b,c,h,w = hrms.shape
    if mtf is None:
        if sensor == 'random':
            mtf = genMTF.genMTF_torch(scale, 'random', b*c, device=hrms.device, anisotropic=anisotropic) 
            
        else:
            mtf = genMTF.genMTF_torch(scale, sensor, c, device=hrms.device, anisotropic=anisotropic) 
            mtf = mtf.unsqueeze(dim=0).repeat(b, 1, 1, 1)
        mtf = mtf.permute(2, 0, 1)
    else:
        kh, kw = mtf.shape[-2:]

    
    kh, kw = mtf.shape[-2:] 

    
    mtf = mtf.view(b*c, 1, kh, kw)
    mslr = hrms.view(1, b*c, h, w)
    mslr = F.pad(mslr, pad=((kh-1)//2, (kh-1)//2, (kw-1)//2, (kw-1)//2), mode='replicate')
    mslr = F.conv2d(mslr, mtf, groups=b*c, stride=1)
    mslr = mslr.view(b,c,h,w)
    mtf = mtf.view(b, c, kh, kw)
    mslr = mslr[:,:,int(scale/2)::scale, int(scale/2)::scale]
    return mslr, mtf
        
def Alpha_Estimation(lrms, pan, scale, type='global'):
    h,w = pan.shape
    c = lrms.shape[-1]

    lrms =  upsample_mat_interp23(lrms)
    lrms = np.concatenate((lrms,np.ones((h, w,1))),axis=2)

    pan = pan.reshape(h*w, 1)
    lrms = lrms.reshape(h*w, c+1)
    if type=='global':
        alpha = np.linalg.lstsq(lrms, pan)
        # alpha = np.dot(pan, lrms).view(b,c+1, h, w)
    elif type == 'local':
        pass
    return alpha

def get_patch(imgdict, scale_dict, patch_size):
    if 'LR' in imgdict.keys():
        ih, iw = imgdict['LR'].shape[-2:]
    else:
        ih, iw = imgdict['HR'].shape[-2:]
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)
    sizedict = {t_key:(ix*t_scale, iy*t_scale, patch_size*t_scale) for t_key, t_scale in scale_dict.items()}
    out_patch = {t_key: imgdict[t_key][:,:,ix:ix+t_psize, iy:iy+t_psize] for t_key, (ix, iy, t_psize) in sizedict.items()}
    return out_patch

def augment(imgdict, batch_size, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    aug_index = random.sample(range(batch_size), random.randrange(1,batch_size))

    def _augment(img, aug_index):
        if hflip: img[aug_index] = img[aug_index].flip(dims=(2,))
        if vflip: img[aug_index] = img[aug_index].flip(dims=(3,))
        if rot90: img[aug_index] = torch.transpose(img[aug_index], 2, 3)
        return img

    return {t_key: _augment(t_img, aug_index) for t_key, t_img in imgdict.items()}

def reduce_mean(x, dim=()):
    # x_shape = x.shape
    x = torch.flatten(x, dim[0], dim[1])
    x = x.mean(dim=dim[0], keepdim=True)
    return x

def reduce_sum(x, dim=()):
    # x_shape = x.shape
    x = torch.flatten(x, dim[0], dim[1])
    x = x.sum(dim=dim[0], keepdim=True)
    return x
    
def modPad(x, mod, dim):
    n = x.shape[dim]
    if n%mod != 0:
        padsize = (n//mod+1)*mod - n
        x = F.pad(x, [0, padsize, 0, 0], mode='replicate')
    return x, padsize
    

def linear_attn(q, k, x, EPS=EPS):
    l1, d1  = q.shape[-2:]
    l2, d2 = x.shape[-2:]
    k = k.transpose(-2, -1)
    if l1*d1*l2+l1*l2*d2<= d2*l2*d1+d2*d1*l1:
        q = q@k
        q = q/(q.sum(dim=-1, keepdim=True)+EPS)
        x = q@x
    else:
        x = q@(k@x)
        q = q@k.sum(dim=-1, keepdim=True) + EPS
        x = x/q
    return x

def linear_attn_new(q, k, x, transblock_idx, img_idx, EPS=EPS, attn_type='Spatial'):
    l1, d1  = q.shape[-2:]
    l2, d2 = x.shape[-2:]
    k = k.transpose(-2, -1)

    q = q@k
    q = q/(q.sum(dim=-1, keepdim=True)+EPS)
    if attn_type == 'Spectral':
        spe_file_name = 'vis_attn_specific_pos/%3d_%2d_%s_dependency'%(img_idx, transblock_idx, attn_type)
        vis_fe(q.contiguous().cpu().numpy(), spe_file_name)
    else:
        np.save('vis_attn_matrix/%03d_%02d_%s_dependency.npy'%(img_idx, transblock_idx, attn_type), q.contiguous().cpu().numpy())
    # np.save(, attn_type), ))
    x = q@x
    return x

def vis_fe(spe_dep, file_name):
    fig, axes = plt.subplots(1, spe_dep.shape[1], figsize=[18, 4])
    for i, ax in enumerate(axes):
        ax.imshow(spe_dep[0,i,:,:])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('%s.png'%file_name, dpi=300)

def expattn(q, k, x):
    k = k.transpose(-2, -1)
    q = q@k
    q = torch.softmax(q)
    x = q@x
    return x

def band_dependent_local_mean_std(img, ksize):
    mean_img = kornia.filters.box_blur(img, (ksize,ksize))
    # biased estimation for variance
    
    local_std = kornia.filters.box_blur(img*img, (ksize, ksize)) - mean_img*mean_img
    # print('img_min:%.6f'%img.min().item())
    # print('std_min:%.6f'%local_std.min().item())
    local_std[local_std<0] = 0
    local_std = torch.sqrt(local_std+EPS)
    return mean_img, local_std

def img_local_mean_std(img:torch.Tensor, ksize):
    mean_kernel = img.new_ones((1, img.shape[1], ksize, ksize))
    mean_kernel = mean_kernel/mean_kernel.numel()
    img = F.pad(img, ((ksize-1)//2,)*4, mode='reflect')
    mean_img = F.conv2d(img, mean_kernel, stride=1)
    # biased estimation for variance
    std_img = F.conv2d(img*img, mean_kernel, stride=1) - mean_img*mean_img
    std_img = torch.sqrt(std_img+EPS)
    return mean_img, std_img

def normalize_max(imgdict:dict, scope: str):
    max_dict={}
    for key, value in imgdict.items():
        if scope == 'band':
            max_dict[key] = EPS + einops.reduce(value, 'b c h w-> b c () ()', 'max')
        elif scope == 'img':
            max_dict[key] = EPS + einops.reduce(value, 'b c h w-> b () () ()', 'max')
        imgdict[key] = value/max_dict[key]
    return imgdict, {'max': max_dict}

def de_normalize_max(imgdict:dict, aux_dat):
    return {key: value*aux_dat['max']['LR'] for key, value in imgdict.items()}

def normalize_minmax(imgdict:dict, scope: str):
    max_dict={}
    min_dict = {}
    for key, value in imgdict.items():
        if scope == 'band':
            max_dict[key] =  EPS + einops.reduce(value, 'b c h w-> b c () ()', 'max')
            min_dict[key] =  einops.reduce(value, 'b c h w-> b c () ()', 'min')
        elif scope == 'img':
            max_dict[key] = EPS + einops.reduce(value, 'b c h w-> b () () ()', 'max')
            min_dict[key] =   einops.reduce(value, 'b c h w-> b () () ()', 'min')
        imgdict[key] = (value-min_dict[key])/(max_dict[key]-min_dict[key])
    return imgdict, {'max': max_dict, 'min': min_dict}

def de_normalize_minmax(imgdict:dict, aux_dat):
    return {key: value*(aux_dat['max']['LR']-aux_dat['min']['LR'])+aux_dat['min']['LR'] for key, value in imgdict.items()}

def normalize_global_mean_std(imgdict:dict, scope:str):
    '''
    input:
        scope: the calculation scope for mean and std
    '''
    mean_dict={}
    std_dict={}
    for key, value in imgdict.items():
        if scope == 'band':
            mean_dict[key] = einops.reduce(value, 'b c h w-> b c () ()', 'mean')
            # std_dict[key] = value.view(value.shape[0],value.shape[1], -1).std(dim=-1).view(value.shape[0], value.shape[1], 1, 1)
            # std_dict[key] += EPS
            std_dict[key] = einops.reduce(value*value, 'b c h w-> b c () ()', 'mean')
            std_dict[key] = torch.sqrt(std_dict[key] - mean_dict[key]*mean_dict[key] + EPS)
        elif scope == 'img':
            mean_dict[key] = einops.reduce(value, 'b c h w-> b () () ()', 'mean')
            # std_dict[key] = value.view(value.shape[0], -1).std(dim=-1).view(value.shape[0], 1, 1, 1)
            # std_dict[key] += EPS
            std_dict[key] = einops.reduce(value*value, 'b c h w-> b () () ()', 'mean')
            std_dict[key] = torch.sqrt(std_dict[key] - mean_dict[key]*mean_dict[key] + EPS)
        imgdict[key] = (value-mean_dict[key])/std_dict[key]
    return imgdict, {'mean':mean_dict, 'std':std_dict}

def normalize_local_mean_std(imgdict:dict, ksize:int, scope:str):
    mean_dict={}
    std_dict={}
    for key, value in imgdict.items():
        if scope.lower() == 'band':
            mean_dict[key], std_dict[key] = band_dependent_local_mean_std(value, ksize)
        elif scope.lower() == 'img':
            mean_dict[key], std_dict[key] = img_local_mean_std(value, ksize)
        imgdict[key] = (value-mean_dict[key])/std_dict[key]
    return imgdict, {'mean':mean_dict, 'std':std_dict}


def de_normalize_mean_std(imgdict:dict, aux_dat):
    local_mean, local_std = aux_dat['mean'], aux_dat['std']
    return {key: value*local_std['LR']+local_mean['LR'] for key, value in imgdict.items()}

def normalize_data(imgdict:dict, norm_type:str, scope:str, ksize:int=9):
    '''
    input:  
        norm_type: one of [max_norm, global_mean_std, BD_local_mean_std, BS_mean_std]
        scope: img or band
    '''
    if norm_type == 'max_norm':
        norm_data, aux_dat = normalize_max(imgdict, scope)
    elif norm_type == 'global_mean_std':
         norm_data, aux_dat = normalize_global_mean_std(imgdict, scope)
    elif norm_type == 'local_mean_std':
        norm_data, aux_dat = normalize_local_mean_std(imgdict, ksize, scope)
    elif norm_type == 'minmax':
        norm_data, aux_dat = normalize_minmax(imgdict, scope)
    else:
        return imgdict, None
        raise TypeError('Cannot Recognize the normalization type of [%s]'%norm_type)
    return norm_data, aux_dat
    
def denormalize(imgdict:dict, aux_data:dict, norm_type:str, scale:float):
    if norm_type == 'max_norm':
        rec_dat = de_normalize_max(imgdict, aux_data)
    elif norm_type == 'global_mean_std':
        rec_dat = de_normalize_mean_std(imgdict, aux_data)
    elif norm_type == 'local_mean_std':
        for key, value in aux_data.items():
            aux_data[key] = {t:F.interpolate(v, scale_factor=scale, mode='nearest') 
                             for t, v in value.items()}
        rec_dat = de_normalize_mean_std(imgdict, aux_data)
        # print('auxdat_mean:%.6f'%aux_data['mean']['LR'].min().item())
        # print('auxdat_std:%.6f'%aux_data['std']['LR'].min().item())
    elif norm_type == 'minmax':
        rec_dat = de_normalize_minmax(imgdict, aux_data)
    else:
        return imgdict
        raise TypeError('Cannot Recognize the normalization type of [%s]'%norm_type)
    return rec_dat