import torch.nn as nn
import torch
import torch.nn.functional as F

def data_norm(ms, pan):
    c = ms.shape[1]
    # img = torch.stack(imgs, dim=0)
    img = torch.cat([ms, pan], dim=1)
    img = (img-img.min())/(img.max()-img.min())
    return img[:,:c,:,:], img[:,c,:,:]




class Net(nn.Module):
    def __init__(self,opt):
        super(Net,self).__init__()
        in_channel = opt['LRdim']

        self.conv1=nn.Conv2d(in_channel+1,64,9,1,4)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(64,32,5,1,2)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(32,in_channel,5,1,2)

    def forward(self,databatch,mask=None):
        ms = databatch['LR']
        pan = databatch['REF']
        # ms, pan = data_norm(ms, pan)
        y = torch.cat([pan, F.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=False)], dim=1) # modify bilinear to bicubic
        x = self.relu1(self.conv1(y))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def loss(self,rec, databatch):
        return F.mse_loss(rec, databatch['GT'])