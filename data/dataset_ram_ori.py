# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023 Lihui Chen
All rights reserved. This work should only be used for nonprofit purposes.

@file        : dataset_ram_ori.py
@Date        : 2023/07/21
@Author      : Lihui Chen
@version     : 1.0
@description : 
@reference   :
'''
from torch import NoneType
import torch.utils.data as data
from urllib3 import Retry
import numpy as np
from . import fileio, preproc, trans_data
import os
from utils.pan_metrics.imresize import imresize
from utils.pan_metrics.MTF import MTF as np_MTF
from utils.pan_metrics.genMTF import genMTF as np_genMTF
from utils.est_srf import est_srf_lsq
from .augmentation.util import im2col
import einops
# from .augmentation.spe_degrade import np_spe_degrade
from .augmentation.crop import dict_random_crop
# from .augmentation.downsample import change_scale
import cv2
from math import ceil

EPS = 1e-7
# from data.trans_data import data2device
# import torch

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def __init__(self, opt, net_arch=NoneType):
        super(LRHRDataset, self).__init__()
        self.opt              = opt
        self.name             = opt['name']
        self.is_train         = ('train' in opt['phase'])
        self.scale            = opt['scaledict']['REF']
        self.repeat           = opt['repeat'] if self.opt['repeat'] else 1
        self.runRange         = opt['run_range']
        self.imgRange         = opt['img_range']
        self.patchSize        = opt['patch_size']
        self.preload          = 'RAM' in opt['setmode']
        self.is_degrade       = opt['is_degrade'] and self.is_train
        self.is_srf           = opt['is_srf'] and self.is_train
        self.choose_band      = opt['choose_band'] and self.is_train
        self.scale_change     = opt['scale_change'] and self.is_train
        self.get_mtf          = opt['get_MTF']
        self.scale_delta      = opt['scale_delta']
        self.rescale_interval = (opt['low_thre'], opt['high_thre'])
        
        ### get lr/hr image paths
        self.scaledict = opt['scaledict']
        self.img_paths = dict()
        for t_key in self.scaledict.keys():
            if self.scale_change and t_key=='LR': #
                self.patchSize = self.patchSize*self.scale
                continue
            tpath                 = os.path.join(opt['data_root'], t_key)
            self.img_paths[t_key] = fileio.get_image_paths(tpath, opt['data_type'])
            
        if self.scale_change:
            self.img_paths['REF_FR'] = fileio.get_image_paths(os.path.join(opt['data_root'], 'REF_FR'), opt['data_type'])
    
        self.data_len = len(self.img_paths[t_key])
        
        print('[online degrade: %s; online srf: %s; scale change: %s; random band: %s]'%(self.is_degrade, self.is_srf, self.scale_change, self.choose_band))

        ### load images to ram
        if self.preload:
            print('Loading images from %s'%opt['data_root'])
            self.imgdict = {t_key:fileio.read_img(t_value, opt['data_type']) 
                            for t_key, t_value in self.img_paths.items()}
            print('===> End Loading [%04d] images <===\n'%self.data_len)
    

    def __getitem__(self, idx):
        ############# load image
        idx = self._get_index(idx)
        pathdict = {t_key:t_value[idx] for t_key, t_value in self.img_paths.items()}
        if self.preload:
            imgbatch_dict = {t_key:t_value[idx] for t_key, t_value in self.imgdict.items()}
        else: 
            imgbatch_dict = fileio.read_img(pathdict, self.opt['data_type'])
            
        ############ if random choose band
        if self.choose_band and np.random.random()<0.5:
              band_idx             = np.random.randint(0, imgbatch_dict['GT'].shape[-1])
            # ref                  = imgbatch_dict.pop('REF')
              imgbatch_dict        = {key: self.choose_dat(value, key, band_idx) for key, value in imgbatch_dict.items()}
            # imgbatch_dict['REF'] = ref
        
        ########### rescal of GSD ################
        if self.scale_change and np.random.random()<self.scale_delta: 
            random_scale = np.random.uniform(self.rescale_interval[0], self.rescale_interval[1]) #
            ############ rescale GT image by MTF with random scale ration #########
            imgbatch_dict['GT'] = np_MTF(imgbatch_dict['GT'], sensor=self.name.split('_')[-1], ratio=random_scale, returnMTF=False) # mtf low-pass filter
            h, w = imgbatch_dict['GT'].shape[:2]
            imgbatch_dict['GT'] = cv2.resize(imgbatch_dict['GT'], (ceil(h*random_scale), ceil(w*random_scale)), interpolation=cv2.INTER_LINEAR_EXACT) # decimation 
            imgbatch_dict['REF'] = imresize(imgbatch_dict['REF_FR'], random_scale*1.0/self.scale)     
        if self.scale_change: imgbatch_dict.pop('REF_FR')
        
        ########## crop patch ########
        if self.is_train: 
            imgbatch_dict = self._get_patch(imgbatch_dict)
            
        ############# MTF degradation
        if self.is_degrade: imgbatch_dict = self.degrade(imgbatch_dict, anisotropic=True)
        mtf = imgbatch_dict.pop('MTF', 'null')
        
        
        ############ if use random srf degradation ############
        # if self.is_srf and np.random.random() < 0.5: 
        #     coeff, _ = est_srf_lsq(imgbatch_dict['GT'], imgbatch_dict['REF'])
        #     alpha = np.random.random()
        #     real_pan = imgbatch_dict['REF']
        #     synthetic_pan = np_spe_degrade(imgbatch_dict['GT'], coeff, False)
        #     synthetic_pan = (synthetic_pan-synthetic_pan.mean())/(synthetic_pan.std()+EPS)*(real_pan.std()+EPS)+real_pan.mean()
        #     imgbatch_dict['REF'] = alpha*synthetic_pan+(1-alpha)*real_pan
            
        ############# np to tensor and then to cuda tensor #######
        imgbatch_dict = trans_data.np2tensor(imgbatch_dict, self.imgRange, self.runRange)
        if self.get_mtf: 
            if isinstance(mtf, str) and mtf=='null':
                mtf = np_genMTF(self.scale, self.name.split('_')[-1], imgbatch_dict['GT'].shape[0])
            mtf = trans_data.np2tensor(mtf, 1.0, 1.0)
            imgbatch_dict['MTF']=mtf
        return (imgbatch_dict, pathdict)
    
    def choose_dat(self, value, key, band_idx):
        if key=='REF':
            return value
        else:
            return value[:,:,band_idx:band_idx+1] if isinstance(band_idx, int) else value[:,:,band_idx]
            
        
    
    def __len__(self):
        return self.data_len * self.repeat if self.is_train else self.data_len

    def _get_index(self, idx):
        return idx % self.data_len if self.is_train else idx

    def _get_patch(self, imgdict):
        imgdict = dict_random_crop(imgdict, self.patchSize, 'np')
        imgdict = preproc.augment(imgdict)
        # lr = common.add_noise(lr, self.opt['noise'])
        return imgdict
    
    def degrade(self, imgdict:dict, spatio_invariant=True, anisotropic=False):
        if spatio_invariant:
            img, mtf = np_MTF(imgdict['GT'], sensor='random', ratio=self.scale, anisotropic=anisotropic)
            imgdict['MTF'] = mtf 
        else:
            H, W, C = imgdict['GT'].shape
            mtf_filter = np_genMTF(self.scale, 'random', H*W*C)
            mtf = mtf_filter
            kh, kw = mtf_filter.shape[:2]
            mtf_filter = mtf_filter.reshape(kh, kw, H*W, C)
            mtf_filter = einops.rearrange(mtf_filter, 'h w N C -> (h w) N C')
            img = np.zeros((H, W, C))
            for idx in range(C):
                tmp_k = mtf_filter.reshape(kh*kw, H*W, C)
                tmp_gt = imgdict['GT'][np.newaxis, :,:,:]
                tmp_gt = im2col(tmp_gt, kh, kw, stride=1, pad=(kh-1)//2, pad_type='edge')
                img = einops.reduce(tmp_k*tmp_gt, 'a b -> b', 'sum').reshape(H, W)
            
        img = img[int(self.scale//2)::self.scale, int(self.scale//2)::self.scale, :]
        imgdict['LR'] = img
        return imgdict
    
    
    