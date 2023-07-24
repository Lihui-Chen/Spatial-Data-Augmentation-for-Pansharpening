import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
# from .common_blocks import conv2d


def conv2d(inCh, outCh, k_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(inCh, outCh, kernel_size=k_size, stride=stride, padding=pad),
        nn.ReLU(inplace=True),
    )

class MS_Encoder(nn.Module):
    def __init__(self):
        super(MS_Encoder, self).__init__()
        self.conv1 = conv2d(1, 64, 3, 1, 1)
        self.conv2 = conv2d(64, 64, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv3 = conv2d(64, 128, 3, 1, 1)
        self.conv4 = conv2d(128, 128, 3, 1, 1)

        self.conv5 = conv2d(128, 256, 3, 1, 1)
        self.conv6 = conv2d(256, 256, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_1 = x + self.conv2(x) #todo:
        x = self.maxpool(x_1)
        x = self.conv3(x)
        x_2 = x + self.conv4(x) #todo:
        x = self.maxpool(x_2)
        x = self.conv5(x)
        x = x + self.conv6(x) #todo:
        return (x_1, x_2, x)


class PAN_Encoder(nn.Module):
    def __init__(self):
        super(PAN_Encoder, self).__init__()
        self.conv1 = conv2d(1, 64, 3, 1, 1)
        self.conv2 = conv2d(64, 64, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv3 = conv2d(64, 64, 3, 1, 1)
        self.conv4 = conv2d(64, 64, 3, 1, 1)

        self.conv5 = conv2d(64, 64, 3, 1, 1)
        self.conv6 = conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_1 = x + self.conv2(x) #todo:
        x = self.maxpool(x_1)
        x = self.conv3(x)
        x_2 = x + self.conv4(x) #todo:
        x = self.maxpool(x_2)
        x = self.conv5(x)
        x = x + self.conv6(x) #todo
        return (x_1, x_2, x)

class LR_Decoer(nn.Module):
    def __init__(self):
        super(LR_Decoer, self).__init__()
        self.fuse1=nn.Sequential(  #todo:
            conv2d(256 + 64, 256, 3, 1, 1),
            conv2d(256, 256, 3, 1, 1)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse2=nn.Sequential(  #todo:
            conv2d(256 + 128 + 64, 128, 3, 1, 1),
            conv2d(128, 128, 3, 1, 1)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse3=nn.Sequential(
            conv2d(128 + 64 + 64, 64, 3, 1, 1),
            conv2d(64, 64, 3, 1, 1)
        )

    def forward(self, ms_list, lrpan_list):
        x = torch.cat((ms_list[2], lrpan_list[2]), dim=1)
        x = self.fuse1(x)
        x = self.up1(x)
        if ms_list[1].shape[-2:] != x.shape[-2:]:
            padh, padw= ms_list[1].shape[-2]- x.shape[-2], ms_list[1].shape[-1]-x.shape[-1]
            x = F.pad(x, (0,padh,0,padw), mode='replicate')
        x = torch.cat((x, ms_list[1], lrpan_list[1]), dim=1)
        x = self.fuse2(x)
        x = self.up2(x)
        if ms_list[0].shape[-2:] != x.shape[-2:]:
            padh, padw= ms_list[0].shape[-2]- x.shape[-2], ms_list[0].shape[-1]-x.shape[-1]
            x = F.pad(x, (0,padh,0,padw), mode='replicate')
        x = torch.cat((x, ms_list[0], lrpan_list[0]), dim=1)
        x=self.fuse3(x)
        return x


class HR_Decoder(nn.Module):
    def __init__(self):
        super(HR_Decoder, self).__init__()
        self.fuse1=nn.Sequential(
            conv2d(64 + 64, 64, 3, 1, 1),
            conv2d(64, 64, 3, 1, 1)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fuse2=nn.Sequential(     # todo:
            conv2d(64 + 64, 64, 3, 1, 1),
            conv2d(64, 64, 3, 1, 1)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fuse3=nn.Sequential(
            conv2d(64 + 64, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1)
        )


    def forward(self, fuse, hrpan_list):
        fuse = torch.cat((fuse, hrpan_list[2]), dim=1)
        fuse = self.fuse1(fuse)
        fuse = self.up1(fuse)
        fuse = torch.cat((fuse, hrpan_list[1]), dim=1)
        fuse = self.fuse2(fuse)
        fuse = self.up2(fuse)
        fuse = torch.cat((fuse, hrpan_list[0]), dim=1)
        fuse = self.fuse3(fuse)
        return fuse

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.ms_encoder = MS_Encoder()
        self.pan_encoder = PAN_Encoder()
        self.lr_decoder = LR_Decoer()
        self.hr_decoder = HR_Decoder()
        self.scale = opt['scale']

    def forward(self, batch_dat, mask=None):
        pan = batch_dat['REF']
        ms = batch_dat['LR']
        lrpan = F.interpolate(pan, scale_factor=(1.0/self.scale), mode='bicubic', align_corners=False)
        ms_list = torch.split(ms, 1, dim=1)
        hr_list = []
        for band in ms_list:
            ms_felist = self.ms_encoder(band)
            lrpan_felist = self.pan_encoder(lrpan)
            fuse = self.lr_decoder(ms_felist, lrpan_felist)
            hrpan_felist = self.pan_encoder(pan)
            fuse = self.hr_decoder(fuse, hrpan_felist)
            hr_list.append(fuse)
        fuse = torch.cat(hr_list, dim=1)
        return fuse
    
    def loss(self, rec, batchdat):
        return F.l1_loss(rec, batchdat['GT'])
