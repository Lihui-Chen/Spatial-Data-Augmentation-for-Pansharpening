#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ArbRPN.py
@Contact :   lihuichen@126.com
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
xxxxxxxxxx        LihuiChen      1.0         None
'''

# import lib


import torch.nn as nn
import torch
import torch.nn.functional as F

# from .loss import make as loss_make
from .common_fn import MTF_Down as torch_MTFDown
from .common_fn import normalize_data, denormalize
from data.augmentation import  crop


class ResBlock(nn.Module)                                                               : 
    def   __init__(self, inFe, outFe, kernel_size=3, stride=1, padding=1, actType=nn.ReLU()): 
        super(ResBlock, self).__init__()
        self.is_linear = False
        if inFe != outFe:
            self.linear = nn.Conv2d(inFe, outFe, 1, 1, 0)
            self.is_linear = True
        self.conv1 = nn.Conv2d(inFe, outFe, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = actType
        self.conv2 = nn.Conv2d(outFe, outFe, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        if self.is_linear:
            x = self.linear(x)
        x = x + res
        return x

class Net(nn.Module):
    def __init__(self, opt=None):
        super(Net, self).__init__()
        hid_dim = 64
        input_dim = 64
        num_resblock = opt['num_res']
        self.num_cycle = opt['num_cycle']
        self.scale = opt['scale']
        self.norm_type = opt['norm_type']
        self.scope = opt['scope']

        self.gt = None

        # self.cut_blur = True

        self.wrapper = nn.Conv2d(1, hid_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(1, input_dim, 3, 1, 1)
        
  

        self.hidden_unit_forward_list = nn.ModuleList()
        self.compress_1 = nn.ModuleList()
        self.hidden_unit_backward_list = nn.ModuleList()
        self.compress_2 = nn.ModuleList()
        
        for _ in range(self.num_cycle):
            self.compress_1.append(nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_1 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_forward_list.append(resblock_1)


            self.compress_2.append(nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_2 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_backward_list.append(resblock_2)

        self.conv2 = nn.Conv2d(hid_dim, 1, 3, 1, 1)
    
    def update_temperature(self,):
        for idx in range(self.num_cycle):
            for res_idx in range(len(self.hidden_unit_backward_list[idx])):
                self.hidden_unit_backward_list[idx][res_idx].update_temperature()
                self.hidden_unit_forward_list[idx][res_idx].update_temperature()
            
    def forward(self, inputbatch, mask=None, is_cat_out=False, is_degrade=False):
        '''
        :param ms: LR ms images
        :param pan: pan images
        :param mask: mask to record the batch size of each band
        :return:
            HR_ms: a list of HR ms images,
        '''
        if is_degrade: ## used for data augmentation by pytorch on CUDA
            mask_matrix = self.batchmask2bandmask(mask)
            band_mask = list((int(tmp.item()) for tmp in mask_matrix.sum(dim=1)))
            gt = inputbatch['GT'].split(1, dim=0)
            pan = inputbatch['REF'].split(1, dim=0)
            lr = []
            new_pan = []
            new_gt = []
            for imgidx, tmpgt in enumerate(gt):
                origt = tmpgt
                tmpgt = tmpgt[:, :band_mask[imgidx], :,:]
                tmppan= pan[imgidx]
                # scale change
                # if scale_change:
                #     tmpgt, tmppan, origt = scale_change.scale_change((tmpgt, tmppan, origt), scale=np.random.uniform(0.25, 2), op_type='torch')
                # random crop
                tmpgt, tmppan, origt = crop.random_crop((tmpgt, tmppan, origt), 64, op_type='torch')
                # random flip and rotations
                # random MTF degradation with spatio-variant MTF
                tmplr, mtf = torch_MTFDown(tmpgt, self.scale, sensor='random', anisotropic=True)
                tmplr = F.interpolate(tmplr, scale_factor=self.scale, mode='bicubic', align_corners=False)
                lr.append(tmplr)
                new_pan.append(tmppan)
                new_gt.append(origt)
            pan = torch.cat(new_pan, dim=0)

        if is_degrade:
            ms = lr 
            self.gt = torch.cat(new_gt, dim=0)
        else:
            ms = inputbatch['LR']
            ms = F.interpolate(ms, scale_factor=self.scale, mode='bicubic', align_corners=False)
            pan = inputbatch['REF'] 
            self.gt = None

     
        
        if mask is None:
            mask = [ms.shape[0] for _ in range(ms.shape[1])]
        self.mask = mask
        
        # B, C, H, W = ms.shape
        if self.norm_type is not None:
            self.aux_dat = []
            pan, _ = normalize_data({'REF':pan}, self.norm_type, self.scope)
            pan = pan['REF']
            if not is_degrade:
                ms = list(ms.split(1, dim=0))
                mask_matrix = self.batchmask2bandmask(mask)
                band_mask = list((int(tmp.item()) for tmp in mask_matrix.sum(dim=1)))
            for idx in range(len(band_mask)):
                tmp_ms = ms[idx][:, :band_mask[idx], ...]
                tmp_ms, tmp_aux = normalize_data({'LR':tmp_ms}, self.norm_type, self.scope)
                self.aux_dat.append(tmp_aux)
                ms[idx] = tmp_ms['LR']
            ms = self.img2batch(ms, mask)
        else:
            ms = ms.split(1, dim=1)
            
   
        pan_state = self.wrapper(pan)
        hidden_state = pan_state
        blur_ms_list = [ms[idx][:mask[idx]] for idx in range(len(ms))]
     

        # the first backward recurrence
        backward_hidden = self.fe_extractor(blur_ms_list, self.conv1, mask)
        backward_hidden = backward_hidden[::-1]
        
        for idx_cycle in range(self.num_cycle):
            ## forward recurrence
            forward_hidden = []
            for idx in range(len(blur_ms_list)):
                hidden_state = hidden_state[:mask[idx]]
                band = torch.cat((backward_hidden[-(idx+1)], hidden_state, pan_state[:mask[idx]]), dim=1)
                band = self.compress_1[idx_cycle](band)
                # tmp_mtf = mtf_fe[idx]
                # hidden_state, _ = self.hidden_unit_forward_list[idx_cycle]((band, tmp_mtf))
                hidden_state= self.hidden_unit_forward_list[idx_cycle](band)
                forward_hidden.append(hidden_state)
                
            ## backward recurrence
            backward_hidden = []
            for idx in range(len(blur_ms_list)):
                start_pan_stat = hidden_state.shape[0]
                hidden_state = torch.cat((hidden_state, pan_state[start_pan_stat:mask[-(idx+1)]]),dim=0)
                band = torch.cat((forward_hidden[-(idx + 1)], hidden_state, pan_state[:mask[-(idx+1)]]), dim=1)
                band = self.compress_2[idx_cycle](band)
                # tmp_mtf = mtf_fe[-(idx+1)]
                # hidden_state, _ = self.hidden_unit_backward_list[idx_cycle]((band, tmp_mtf))
                hidden_state = self.hidden_unit_backward_list[idx_cycle](band)
                backward_hidden.append(hidden_state)

        # the last forward recurrence
        HR_ms = []
        HR_ms = self.fe_extractor(backward_hidden[::-1], self.conv2, mask, if_denorm=(self.norm_type is not None))
        HR_ms = [HR_ms[idx]+blur_ms_list[idx] for idx in range(len(HR_ms))]
            
        # return HR_ms if not is_cat_out else (HR_ms, torch.cat(HR_ms, dim=1))
        if self.norm_type is not None:
            HR_ms = self.batch2img(HR_ms, band_mask)
            for idx, tmp in enumerate(HR_ms):
                HR_ms[idx] = denormalize({'OUT':tmp}, self.aux_dat[idx], self.norm_type, self.scale)['OUT']
            HR_ms = self.img2batch(HR_ms, mask)
        return torch.cat(HR_ms, dim=1) if len(set(mask))==1 else HR_ms

    
    def loss(self, ms, databatch):
        HR = databatch['GT'] if self.gt is None else self.gt
        
        ms = torch.split(ms, 1, dim=1) if isinstance(ms, torch.Tensor) else ms 
        HR = torch.split(HR, 1, dim=1) if isinstance(HR, torch.Tensor) else HR
        
        HR = [HR[idx][:self.mask[idx]] for idx in range(len(self.mask))]
        ms = torch.cat(ms, dim=0)
        HR = torch.cat(HR, dim=0)

        loss = torch.sum(torch.abs(ms-HR))/ms.numel()
    
        return loss
    

    
    def get_loss(self, rec=None, databatch:dict=None):
        if self.loss_name in ('l1', 'l2', 'smoothL1'):
            self.loss(rec, databatch['GT'])
        else:
            self.loss(rec, databatch)
            
    def fe_extractor(self, x, conv, mask, if_norm=False, if_denorm=False):    
        x = [x[idx][:mask[idx]] for idx in range(len(x))]
        x = torch.cat(x, dim=0)
        x = conv(x)
        x = x.split(mask, dim=0)
        return x
    
    def batchmask2bandmask(self, batchmask):
        mask_matrix = torch.zeros((batchmask[0], len(batchmask)))
        for idx, tmp in enumerate(batchmask):
            mask_matrix[:tmp, idx:idx+1] = mask_matrix[:tmp, idx:idx+1] + torch.ones((tmp, 1))
        return mask_matrix
    
    def batch2img(self, batchlist:list, bandmask):
        new_list = []
        for idx, tmp in enumerate(batchlist):
            new_list.append(tmp.split(1, dim=0))
            if idx==0:
                imgcount = tmp.shape[0]
        img_list = []
        for imgidx in range(imgcount):
            tmp_list = []
            for bandidx in range(bandmask[imgidx]):
                tmp_list.append(new_list[bandidx][imgidx])
            rec_img = torch.cat(tmp_list, dim=1)
            # rec_img = denormalize({'OUT':rec_img}, self.aux_dat[imgidx], self.norm_type)
            img_list.append(rec_img)
        return img_list
    
    def img2batch(self, imglist, batchmask):
        new_list = []
        for imgidx in range(len(imglist)):
            new_list.append(imglist[imgidx].split(1, dim=1))
            if imgidx == 0: 
                maxband_numb = imglist[imgidx].shape[1]
            
        batch_list = []
        for bandidx in range(maxband_numb):
            tmp_batch_bands = []
            for imgidx in range(batchmask[bandidx]):
                tmp_batch_bands.append(new_list[imgidx][bandidx])
            tmp_batch = torch.cat(tmp_batch_bands, dim=0)
            batch_list.append(tmp_batch)
        return batch_list        
        