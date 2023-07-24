"""
Author: LihuiChen
E-mail: lihuichen@126.com
Note: The metrics for reduced-rolution is the same with the matlat codes opened by [Vivone20]. 
      Metrics for full-resolution have a little different results from the codes opened by [Vivone20].

Refercence: PansharpeningToolver1.3 and Pansharpening Toolbox for Distribution

Pansharpening metrics: The same implementation of CC, SAM, ERGAS, Q2n as the one in Matlab codes publised by:
    [Vivone15]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, 
                "A Critical Comparison Among Pansharpening Algorithms", IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565�2586, May 2015.
    [Vivone20]  G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, 
                "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and 
                emerging pansharpening methods",IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
"""
from scipy.ndimage import sobel
import numpy as np
# from scipy import signal, ndimage, misc
import cv2
from numpy.linalg import norm
# from PIL import Image

EPS = 1e-7

##########################################################
# Full Reference metrics for Reduced Resolution Assesment
##########################################################

def SAM(ms,ps,degs = True):
    result = np.double(ps)
    target = np.double(ms)
    
    if result.shape != target.shape: raise ValueError('Result and target arrays must have the same shape!')

    bands = target.shape[2]
    rnorm = (result ** 2).sum(axis=2)
    tnorm = (target ** 2).sum(axis=2)
    
    dotprod = (result * target).sum(axis=2)
    rnorm = np.sqrt(rnorm*tnorm)
    tnorm = rnorm
    rnorm = np.maximum(rnorm, EPS)
    cosines = (dotprod / rnorm)
    sam2d = np.arccos(cosines)
    if degs:
        sam2d = np.rad2deg(sam2d)
    
    rnorm   = tnorm[tnorm>0]
    dotprod = dotprod[tnorm>0]
    sam     = np.arccos(dotprod/rnorm).mean()
    
    if degs:
        sam = np.rad2deg(sam)
    # sam2d[np.invert(np.isfinite(sam2d))] = 0.  # arccos(1.) -> NaN
    return sam


def CC(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')

def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    return  (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))


def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0 
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq)!=0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)


def Q_AVE(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')

def ERGAS(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mean_real = mean_real**2
        mse = np.mean((img_fake_ - img_real_)**2)
        mean_real = np.maximum(mean_real, EPS)
        return 100.0 / scale * np.sqrt(mse / mean_real)
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        means_real = means_real**2
        means_real = np.maximum(means_real, EPS)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100.0 / scale * np.sqrt((mses/means_real).mean())
    else:
        raise ValueError('Wrong input image dimensions.')
    



##########################################################
# Using Gaussian filter matched MTF to degrade HRMS images
##########################################################


if __name__ == '__main__':

# import numpy as np
# import os
# ArbRPN_dir = '/home/ser606/Documents/LihuiChen/ArbRPN_20200916/results/SR/RNN_RESIDUAL_BI_PAN_FB_MASK/QB-FIX-4/x4/'
# GT_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/HRMS_npy'
# LRMS_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/LRMS_npy'
# PAN_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/LRPAN_npy'
# # save_root = '/home/ser606/Documents/LihuiChen/compare/extra/reduced_resolution/test'

# ArbRPN_files = os.listdir(ArbRPN_dir)
# ArbRPN_files.sort()
# GT_files = os.listdir(GT_dir)
# GT_files.sort()
# LRMS_files = os.listdir(LRMS_dir)
# LRMS_files.sort()
# PAN_files = os.listdir((PAN_dir))
# PAN_files.sort()
# cc = []
# sam = []
# ergas = []
# q_ave = []
# q2n = []
# for i in range(len(ArbRPN_files)):
#     ps = np.load(os.path.join(ArbRPN_dir, ArbRPN_files[i]))
#     ps = ps.astype(np.float)
#     gt = np.load(os.path.join(GT_dir, GT_files[i])).astype(np.float)
#     pan = np.load(os.path.join(PAN_dir, PAN_files[i])).astype(np.float)
#     # print('%s  ||  %s \n'%(ArbRPN_files[i], PAN_files[i]))
#     # print((ps.dtype))
#     cc.append(CC(ps, gt))
#     sam.append(SAM(gt, ps))
#     ergas.append(ERGAS(ps, gt))
#     q_ave.append(Q_AVE(ps, gt))
#     q2n.append(Q2n(gt, ps, 32, 32))

# print(mean(q2n))

    # import os
    # TFNET_dir = '/home/ser606/Documents/LihuiChen/ArbRPN_20200916/results/SR/PARABIRNN/QB-Vanilla-BiRNN-FR/x4'
    # LRMS_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MS_full_resolution/MS_npy'
    # PAN_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/PAN_full_resolution/PAN_npy'
    # # save_root = '/home/ser606/Documents/LihuiChen/compare/extra/full_resolution/test'ArbRPN_files = os.listdir(ArbRPN_dir)
    # TFNET_files = os.listdir(TFNET_dir)
    # TFNET_files.sort()
    # # GT_files = os.listdir(GT_dir)
    # # GT_files.sort()
    # LRMS_files = os.listdir(LRMS_dir)
    # LRMS_files.sort()
    # PAN_files = os.listdir((PAN_dir))
    # PAN_files.sort()
    # Dl_results = []
    # Ds_results = []
    # Qnr_results = []
    # for i in range(len(TFNET_files)):
    #     print('processing the %d-th image.\n'%i)
    #     ps = np.load(os.path.join(TFNET_dir, TFNET_files[i]))
    #     ps = ps.astype(np.float)
    #     lrms = np.load(os.path.join(LRMS_dir, LRMS_files[i])).astype(np.float)
    #     pan = np.load(os.path.join(PAN_dir, PAN_files[i])).astype(np.float)
    #     # print('%s  ||  %s \n'%(ArbRPN_files[i], PAN_files[i]))
    #     # print((ps.dtype))
    #     msexp = upsample_mat_interp23(lrms, 4)
    #     dl, ds, hqnr = HQNR(ps, lrms, msexp, pan, 32, 'QB', 4)
    #     Dl_results.append(dl)
    #     Ds_results.append(ds)
    #     Qnr_results.append(hqnr)
    # print(sum(Qnr_results)/len(Qnr_results))
    a = np.random.randn(240, 240, 128)
    b = np.random.randn(240, 240, 128)
    Q2n(a,b)
