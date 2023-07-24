'''
@File    :   FE
@Contact :   634350973@qq.com
@License :   None

@Modify Time      @Author    @Version    @Desciption
-------------     --------    --------    -----------
2022/2/12   LihuiChen      1.0           estimate a shared psr for all bands in MS images

Reference:
    [1]G. Vivone et al., “Pansharpening Based on Semiblind Deconvolution,” IEEE Trans. Geosci. Remote Sensing, vol. 53, no. 4, pp. 1997–2010, Apr. 2015, doi: 10.1109/TGRS.2014.2351754.
'''
import numpy as np
import math
from scipy import ndimage
from pan_metrics.estimate_srf import estimateAlpha
import pywt


def FE(I_MS: np.ndarray, I_PAN: np.ndarray, Resize_fact: int, tap, lmd, mu, th, num_iter, filtername):

    if tap % 2 == 0:
        sum_tap = 0
    else:
        sum_tap = 1
    tap = math.floor(tap/2)

    R_SIZE, C_SIZE = I_PAN.shape

    if filtername == 'Naive2':
        gv = np.zeros(2, 1)
        gv[0, 0] = -1
        gv[1, 0] = 1

        gh = np.zeros(1, 2)
        gh[0, 0] = -1
        gh[0, 1] = 1
    elif filtername == 'Naive3':
        gv = np.zeros(3, 1)
        gv[0, 0] = -1
        gv[2, 0] = 1

        gh = np.zeros(1, 3)
        gh[0, 0] = -1
        gh[0, 2] = 1
    elif filtername == 'Basic':
        gv = np.zeros(2, 2)
        gv[0, :] = -1
        gv[1, :] = 1

        gh = np.zeros(2, 2)
        gh[:, 0] = -1
        gh[:, 1] = 1
    elif filtername ==  'Prewitt':
        gv = np.zeros(3, 3)
        gv[0, : ] = -1
        gv[2, : ] = 1

        gh = np.zeros(3, 3)
        gh[: , 0] = -1
        gh[: , 2] = 1
    elif filtername ==  'Sobel':
            # gv = np.zeros(3, 3);
            # gv[1, 1] = -1; gv(1, 2) = -2; gv(1, 3) = -1
            # gv[3, 1) = +1; gv(3, 2) = +2; gv(3, 3) = +1

            # gh = zeros(3, 3);
            # gh(1, 1) = -1; gh(2, 1) = -2; gh(3, 1) = -1
            # gh(1, 3) = +1; gh(2, 3) = +2; gh(3, 3) = +1
            gv = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
            
            gh = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    else:
            # gv = zeros(2, 2);
            # gv(1, : ) = -1;
            # gv(2, : ) = 1;
            # gh = zeros(2, 2);
            # gh(: , 1) = -1;
            # gh(: , 2) = 1;
            gv = np.array([[-1, -1],
                           [1, 1]])
            gh = np.array([[-1, 1],
                           [-1, 1]])
            

    gvf = np.fft.fft2(gv, R_SIZE, C_SIZE)
    ghf = np.fft.fft2(gh, R_SIZE, C_SIZE)

    gvfc = np.conj(gvf)
    ghfc = np.conj(ghf)

    gvf2 = gvfc*gvf
    ghf2 = ghfc*ghf

    gf2sum = gvf2 + ghf2

    H_E = I_PAN.astype(np.double)

    for jj in range(num_iter):
    #     Filter PAN to estimate alpha set
        if jj == 1:
            PAN_LP = LPfilter(H_E, Resize_fact)
        else:
            PAN_LP = ndimage.filters.correlate(H_E, PSF_l,'replicate')

        # % %% Estimate alpha
        tmpMS = np.concatenate((I_MS, np.ones(I_MS.shape[:2]+[1])), axis=-1)
        alpha = estimateAlpha(tmpMS,PAN_LP,'global')
        
        It_E = (tmpMS*alpha[np.newaxis, np.newaxis, :]).sum(axis=2); 

        #  Edge taper
        H_E = edgetaper(H_E, np.ones(tap,tap)/((tap)**2))
        It_E = edgetaper(It_E, np.ones(tap,tap)/((tap)**2))

        #  Filter Estimation
        PSF = np.real(np.fft.fftshift(np.fft.ifft2(np.conj(np.fft.fft2(H_E))*np.fft.fft2(It_E)/(abs(np.fft.fft2(H_E))**2 + lmd + mu * gf2sum ))))

        #  Thresholding
        PSF[PSF < th] = 0

        #  Cut using the support dimension and center
        _, maxIndex = max(PSF[:])
        [rm, cm] = ind2sub(PSF.shape, maxIndex)
        PSF_l = PSF[rm - tap : rm + tap - 1 + sum_tap, cm - tap : cm + tap - 1 + sum_tap]
        PSF_l = PSF_l / PSF_l.sum()

def edgetaper(img, psf):
    blur_img = ndimage.filters.correlate(img, psf, mode='nearest')
  

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype(np.int32) / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def  LPfilter(HRPan,Resize_fact):
    h=np.array([1, 4, 6, 4, 1 ])/16
    g=np.array([0, 0, 1, 0, 0 ])-h
    htilde=np.array([ 1, 4, 6, 4, 1])/16
    gtilde=[ 0, 0, 1, 0, 0 ]+htilde
    
    h=math.sqrt(2)*h
    g=math.sqrt(2)*g
    htilde=math.sqrt(2)*htilde
    gtilde=math.sqrt(2)*gtilde
    
    WF=(h,g,htilde,gtilde)
    Levels = math.ceil(math.log2(Resize_fact))

    WT = pywt.swt2(HRPan, WF, Levels)

    for ii in range(1, len(WT.dec)):
        WT[ii] = np.zeros(WT[ii].shape)

    HRPanLP = pywt.iswt2(WT, WF)
    return HRPanLP

