# from scipy import signal
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


##################
'''
    The following codes is copied from https://github.com/yuanjunchai/IKC/blob/b1103bda95/codes/utils/util.py
    @Apache-2.0 License 
'''
# def isogkern(kernlen, std): 
# gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
# gkern2d = np.outer(gkern1d, gkern1d)
# gkern2d = gkern2d/np.sum(gkern2d)
#     return gkern2d


# def anisogkern(kernlen, std1, std2, angle): 
# gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
# gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
# gkern2d   = np.outer(gkern1d_1, gkern1d_2)
# gkern2d   = gkern2d/np.sum(gkern2d)
#     return gkern2d


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k] # PCA matrix

def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma

def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False, ifNorm=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    if ifNorm: kernel = kernel / np.sum(kernel)
    return torch.FloatTensor(kernel) if tensor else kernel


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)


def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = stable_gaussian_kernel(l=l, sig=sig, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def b_GPUVar_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_view


def b_CPUVar_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_v[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_v = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_v


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, random, batch, tensor=False):
        if random == True: #random kernel
            return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate,
                                       scaling=self.scaling, tensor=tensor)
        else: #stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor)


class PCAEncoder(object):
    def __init__(self, weight, cuda=False):
        self.weight = weight #[l^2, k]
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size() #[B, l, l]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))
