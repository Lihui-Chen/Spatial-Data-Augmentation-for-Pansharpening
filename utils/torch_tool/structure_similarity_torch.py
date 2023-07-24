import torch
import kornia

def cal_similarity(M_, hr_pan):
    '''
    inputs:
        M_ and hr_pan should have the same size of [B, C, H, W]  
    return:
        S: the similarity between M_ and hr_pan by local structure similarity 
    '''
    mean_filter = kornia.filters.BoxBlur((31,31))
    e = 1e-10
    r = 4
    a = 1 
    mean_M_ = mean_filter(M_)
    mean_P =  mean_filter(hr_pan) 
    mean_M_xP = mean_filter(M_*hr_pan)
    cov_M_xP = mean_M_xP - mean_M_*mean_P
    mean_M_xM_ = mean_filter(M_*M_)
    std_M_ = torch.sqrt(torch.abs(mean_M_xM_ - mean_M_*mean_M_) + e)
    mean_PxP = mean_filter(hr_pan*hr_pan)
    std_P = torch.sqrt(torch.abs(mean_PxP - mean_P*mean_P) + e)
    corr_M_xP = cov_M_xP / (std_M_*std_P)
    S = corr_M_xP**r
    return S