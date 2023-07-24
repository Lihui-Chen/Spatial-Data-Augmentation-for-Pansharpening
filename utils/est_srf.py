import numpy as np
from scipy.optimize import nnls
import torch

def est_srf_nnls(lrms, pan, regType='global', bias=False):
    H,W = pan.shape[:2]
    h, w, c = lrms.shape[:3]
    assert h==H and w==W, 'the size of the gt image and the ref image is different.'
    pan = pan.reshape(h*w, -1)
    lrms = lrms.reshape(H*W, c)
    if bias: lrms = np.concatenate((lrms,np.ones(H*W, 1)), axis=1)
    if regType=='global':
        alpha = []
        resnorm = []
        for idxband in range(pan.shape[-1]):
            tmpalpha, tmpresnorm = nnls(lrms, pan[:,idxband])
            alpha.append(tmpalpha)
            resnorm.append(tmpresnorm)
        alpha, resnorm = np.array(alpha), np.array(resnorm)
    elif regType == 'local':
        pass
    return alpha, resnorm

def est_srf_lsq(lrms, pan, regType='global', bias=False):
    H,W = pan.shape[:2]
    h, w, c = lrms.shape[:3]
    assert h==H and w==W, 'the size of the gt image and the ref image is different.'
    pan = pan.reshape(h*w, -1)
    lrms = lrms.reshape(H*W, c)
    if bias: lrms = np.concatenate((lrms,np.ones(H*W, 1)), axis=1)
    if regType=='global':
        alpha = []
        resnorm = []
        for idxband in range(pan.shape[-1]):
            tmpalpha, tmpresnorm, _, _ = np.linalg.lstsq(lrms, pan[:,idxband], rcond=-1)
            alpha.append(tmpalpha)
            resnorm.append(tmpresnorm)
        alpha, resnorm = np.array(alpha), np.array(resnorm)
    elif regType == 'local':
        pass
    return alpha, resnorm

def est_srf_torch(lr_ms, pan, bias=False):
    B, C, h, w = lr_ms.shape
    refC = pan.shape[1]
    pan = pan.view(B, refC, -1)
    lr_ms_band = lr_ms.view(B, C, -1)
    if bias:
        lr_ms_band = torch.cat((lr_ms_band, torch.ones_like(lr_ms_band[:, 0:1,:])), dim=1)
    # Bx(C+1)xrefC
    sol = torch.linalg.lstsq(lr_ms_band.transpose(1, 2), pan.transpose(1, 2))
    sol = sol.solution.transpose(1, 2)
    return sol 