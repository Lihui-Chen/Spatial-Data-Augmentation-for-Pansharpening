import matplotlib.pyplot as plt
import numpy as np
import torch

def hist_line_stretch(img, nbins, bound=[0.02, 0.98]):
    def _line_strectch(img, nbins):
        nbins = int(nbins)
        # imgrange = [int(img.min()), np.ceil(img.max())]
        hist1, bins1 = np.histogram(img, bins=nbins, density=False)
        hist1 = hist1/img.size
        cumhist = np.cumsum(hist1)
        lowThreshold= np.where(cumhist>=bound[0])[0][0]
        lowThreshold= bins1[lowThreshold]
        highThreshold = np.where(cumhist>=bound[1])[0][0]
        highThreshold = bins1[highThreshold]
        
        img[np.where(img<lowThreshold)] = lowThreshold
        img[np.where(img>highThreshold)] = highThreshold
        img = (img-lowThreshold)/(highThreshold-lowThreshold+np.finfo(np.float).eps)
        return img
    if img.ndim>2:
        for i in range(img.shape[2]):
            img[:,:,i] = _line_strectch(img[:,:,i].squeeze(), nbins)
    else:
        img = _line_strectch(img, nbins)
    return img

def vis_weights_curves(input_tensor, ):
    in_ch, out_ch, w, h = input_tensor.shape
    input_tensor = input_tensor.view(in_ch, out_ch)


    # fig, ax = plt.subplots(nrows=2, ncols=(out_ch+1)//2)
    fig = plt.figure() 
    # # for i in range(out_ch):
    # #     tmp_weights = input_tensor[:,i]
    #     tmp_weights = tmp_weights.cpu().numpy()
    #     ax[0].plot(tmp_weights)
    tmp_weights = input_tensor[0]
    tmp_weights = tmp_weights.cpu().numpy()
    plt.plot(tmp_weights)
    plt.show()
    plt.close()
    return fig
        
def linstretch(hr, sr, ):
    c, h, w = hr.shape
    for i in range(c):
        max_pix, min_pix = hr[i].max(), hr[i].min()
        data = torch.stack((hr[i], sr[i]), dim=0)
        data[data<min_pix] = min_pix
        data[data>max_pix] = max_pix
        data = (data-min_pix)/(max_pix-min_pix)
        hr[i], sr[i] = torch.split(data, 1, dim=0)

    return hr, sr