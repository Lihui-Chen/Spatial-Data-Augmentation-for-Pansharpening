import torch
import numpy as np
import torch.nn.functional as F
import random
from utils.pan_metrics.tools import gaussian2d, fir_filter_wind, kaiser2d


def GNyq2win(GNyq, scale, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    # fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)


def MTF_Down(hrms:torch.Tensor, scale, sensor=None, mtf=None):
    b, c, h, w = hrms.shape
    if mtf is None:
        if sensor == 'random':
            GNyq = np.random.normal(loc=0.3, scale=0.03, size=b*c)
        elif sensor == 'QB':
            GNyq = [0.34, 0.32, 0.30, 0.22]*b   # Band Order: B,G,R,NIR
        elif sensor == 'IK':
            GNyq = [0.26, 0.28, 0.29, 0.28]*b    # Band Order: B,G,R,NIR
        elif sensor == 'GE' or sensor == 'WV4':
            GNyq = [0.23, 0.23, 0.23, 0.23]*b    # Band Order: B,G,R,NIR
        elif sensor == 'WV3':
            GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]*b
        elif sensor == 'WV2':
            GNyq = ([0.35]*7+[0.27])*b
        else:
            GNyq = [0.3]*c*b
        mtf = [torch.from_numpy(GNyq2win(GNyq=tmp, scale=scale)).type_as(
            hrms).to(hrms.device) for tmp in GNyq]
        kh, kw = mtf[0].shape
        mtf = torch.stack(mtf, dim=0)

    kh, kw = mtf.shape[-2:]
    # ms_lr = torch.zeros_like(hrms)
    # pad_hrms = F.pad(hrms, pad=((kh-1)//2, (kh-1)//2, (kw-1)//2, (kw-1)//2), mode='replicate')
    # for i in range(b):
    #     for j in range(c):
    #         tmp_hr = pad_hrms[i:i+1, j:j+1,:,:]
    #         tmp_mtf = mtf[i:i+1, j:j+1,:,:]
    #         ms_lr[i, j,:,:] = F.conv2d(tmp_hr, tmp_mtf, stride=1).squeeze()

    mtf = mtf.view(b*c, 1, kh, kw)
    mslr = hrms.view(1, b*c, h, w)
    mslr = F.pad(mslr, pad=((kh-1)//2, (kh-1)//2,
                 (kw-1)//2, (kw-1)//2), mode='replicate')
    mslr = F.conv2d(mslr, mtf, groups=b*c, stride=1)
    mslr = mslr.view(b, c, h, w)
    mtf = mtf.view(b, c, kh, kw).contiguous()
    mslr = mslr[:, :, int(scale/2)::scale, int(scale/2)::scale].contiguous()
    return mslr, mtf

def get_patch(imgdict:dict, scale_dict:dict, patch_size:int):
    key = filter(lambda x: scale_dict[x]==1, scale_dict.keys())
    assert key is not None, 'The scale_dict has no key with value eqaul to 1.'
    iw, ih = imgdict[key[0]].shape
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)
    sizedict = {t_key:(ix*t_scale, iy*t_scale, patch_size*t_scale) for t_key, t_scale in scale_dict.items()}
    out_patch = {t_key: imgdict[t_key][:,:,ix:ix+t_psize, iy:iy+t_psize] for t_key, (ix, iy, t_psize) in sizedict.items()}
    return out_patch