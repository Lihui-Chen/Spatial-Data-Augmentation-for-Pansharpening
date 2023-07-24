from matplotlib.pyplot import axis
import numpy as np
from scipy.optimize import nnls

def estimateAlpha(lrms, pan, regType='vanilla'):
    H,W = pan.shape[:2]
    h, w, c = lrms.shape[:3]
    assert h==H and w==W, 'the size of the MS image and the PAN image is different.'
    pan = pan.reshape(H*W, -1)
    lrms = lrms.reshape(h*w, c)
    if regType=='nnls':
        alpha = []
        resnorm = []
        for idxband in range(pan.shape[-1]):
            tmpalpha, tmpresnorm = nnls(lrms, pan[:,idxband])
            alpha.append(tmpalpha)
            resnorm.append(tmpresnorm)
        alpha, resnorm = np.array(alpha)/alpha.sum(axis=1), np.array(resnorm)
    if regType=='vanilla':
        alpha = []
        resnorm = []
        for idxband in range(pan.shape[-1]):
            tmpalpha, tmpresnorm = np.linalg.lstsq(lrms, pan[:,idxband])
            alpha.append(tmpalpha)
            resnorm.append(tmpresnorm)
        alpha, resnorm = np.array(alpha)/alpha.sum(axis=1), np.array(resnorm)
    elif regType == 'local':
        pass
    return alpha, resnorm