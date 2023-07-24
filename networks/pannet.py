import torch.nn as nn
import torch
import torch.nn.functional as F


class AveFilter(nn.Module):
    def __init__(self):
        super(AveFilter, self).__init__()
        ave_kernel = (torch.ones(5, 5) / 25).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=ave_kernel, requires_grad=False)

    def forward(self, x):
        x_list = torch.split(x, 1, dim=1)

        filter_x = [F.conv2d(bands, self.weight, padding=2) for bands in x_list]
        filter_x = torch.cat(filter_x, dim=1)
        return filter_x


class ResBlock(nn.Module):
    def __init__(self, inFe, outFe):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inFe, outFe, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outFe, outFe, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


class Net(nn.Module):
    def __init__(self,opt, inCh=4, numFe=32):
        super(Net, self).__init__()
        in_channel = opt['LRdim']
        refdim = opt['REFdim']

        self.ave_filter = AveFilter()
        self.Tconv = nn.ConvTranspose2d(in_channel, in_channel, 8, 4, 2)
        self.conv1 = nn.Conv2d(in_channel+refdim, numFe, 3, 1, 1)
        self.res_block = nn.Sequential(*[
            ResBlock(numFe, numFe) for _ in range(4)
        ])
        self.convOut = nn.Conv2d(numFe, in_channel, 3, 1, 1)

    def forward(self, databatch, mask=None):
        ms = databatch['LR']
        pan = databatch['REF']
        high_pass_ms = ms - self.ave_filter(ms)

        high_pass_ms = self.Tconv(high_pass_ms)
        pan = pan - self.ave_filter(pan)

        x = torch.cat([high_pass_ms, pan], dim=1)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convOut(x)

        x = x + F.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=False)

        return x
    
    def loss(self, rec, databatch):
        return F.mse_loss(rec, databatch['GT'])
