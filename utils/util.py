import os
import math
from tkinter import image_names
from scipy.stats import pearsonr
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from .metrics import SAM, ERGAS
from .pan_metrics import q2n, HQNR, interp23
# from skimage.measure import compare_psnr
# from skimage.metrics import peak_signal_noise_ratio




####################
# image convert
####################


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)




###Pan-sharpening#####
def pan_calc_metrics_rr(PS, GT, scale, img_range):
    GT = np.array(GT).astype(np.float)/img_range
    PS = np.array(PS).astype(np.float)/img_range
    # RMSE = (GT - PS)/img_range
    # RMSE = np.sqrt((RMSE*RMSE).mean())
    # cc = CC(GT,PS)
    sam = SAM(GT, PS)
    ergas = ERGAS(PS, GT, scale=scale)
    # Qave = Q_AVE(GT, PS)
    # scc = sCC(GT, PS)
    q2n_value, _ = q2n.q2n(GT,PS, 32, 32)
    # return {'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n, 'CC': cc, 'RMSE':RMSE}
    return {'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n_value}

def pan_calc_metrics_all(databatch, scale, img_range, FR=False):
    PS = databatch['OUT'].astype(np.double)
    if not FR:
        # GT = np.array(GT).astype(np.double)/img_range
        # PS = np.array(PS).astype(np.double)/img_range
        # RMSE = (GT - PS)/img_range
        # RMSE = np.sqrt((RMSE*RMSE).mean())
        # cc = CC(GT,PS)
        GT = databatch['GT'].astype(np.double)/img_range
        PS = PS/img_range
        sam = SAM(GT, PS)
        # ergas = round(ERGAS2(PS, GT, scale=scale), 4)
        ergas = ERGAS(GT, PS, scale=scale)
        
        psnr = mPSNR(GT, PS)
        # Qave = Q_AVE(GT, PS)
        # scc = sCC(GT, PS)
        q2n_value, _ = q2n.q2n(GT,PS, 32, 32)
        # return {'PSNR': psnr, 'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n_value}
        rlt = {'PSNR':psnr, 'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n_value}
    else:
        I_MS_LR = databatch['LR'].astype(np.double)
        I_MS = interp23.interp23(I_MS_LR, scale).astype(np.double)
        I_PAN = databatch['REF'].astype(np.double)
        sensor = databatch.pop('MTF') if databatch.get('MTF') is not None else databatch['SENSOR']
        HQNR_index, D_lambda, D_S = HQNR.HQNR(PS,I_MS_LR,I_MS,I_PAN, 32, sensor,scale)
        rlt = {'D_lambda':D_lambda, 'D_S':D_S, 'HQNR':HQNR_index}

    return rlt
    

def mPSNR(ref, tar):
    """ 
    The same implementation as Naoto's toolbox for HSMS image fusion.
    """
    if ref.ndim>2:
        bands = ref.shape[2]
        ref = ref.reshape(-1,bands)
        tar = tar.reshape(-1,bands)
        msr = ((ref-tar)**2).mean(axis=0)
        # max2 = max(ref,[],1).^2;
        max2 = ref.max(axis=0)**2
        psnrall = np.zeros_like(max2)
        if (msr==0).any():
            return float('inf')
        # psnrall[msr==0] = float('inf')
            # return float('inf')
        psnrall = 10*np.log10(max2/msr)
        # out.ave = mean(psnrall); 
        return psnrall.mean()
    else:
        max2 = ref.max()
        msr = ((ref-tar)**2).mean()
        return 10*np.log10(max2/msr)
    
####################
# metric
####################
def calc_metrics_(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    

def count_width(s, align_zh):
    s = str(s)

    count = 0
    for ch in s:
        if align_zh and u'\u4e00' <= ch <= u'\u9fff':  # 中文占两格
            count += 2
        else:
            count += 1

    return count

def print_dict_to_md_table(dict):
    columns, rows = [], []
    for key, value in dict.times():
            columns +=  [key]
            rows += [value]
    print_to_markdwon_table(columns, [rows])
    
def print_to_markdwon_table(column, rows, align_zh = False):

    widths = []
    column_str = ""
    separate = "----"
    separate_str = ""
    for ci, cname  in enumerate(column):
        cw = count_width(cname, align_zh)
        for row in rows:
            item = row[ci]

            if count_width(item, align_zh) > cw:
                cw = count_width(item, align_zh)

        widths.append(cw)

        delete_count = count_width(cname, align_zh) - count_width(cname, False)

        column_str += f'|{cname:^{cw-delete_count+2}}'
        separate_str += f'|{separate:^{cw+2}}'

    column_str += "|"
    separate_str += "|"

    print(column_str)
    print(separate_str)

    for ri, row in enumerate(rows):
        row_str = ""
        for ci, item in enumerate(row):
            cw = widths[ci]

            delete_count = count_width(item, align_zh) - count_width(item, False)
            row_str += f'|{item:^{cw-delete_count+2}}'

        row_str += "|"
        print(row_str)
