import torch

def esrf(hr_pan, lr_ms):
    down_hr_pan = torch.nn.functional.interpolate(hr_pan, size=lr_ms.shape[-2:], mode='area')
    B, C, h, w = lr_ms.shape
    refC = hr_pan.shape[1]
    down_hr_pan = down_hr_pan.view(B, refC, -1)

    # down_hr_pan_array = down_hr_pan.view(B, refC, -1)
    lr_ms_band = lr_ms.view(B, C, -1)
    lr_ms_band = torch.cat((torch.ones_like(lr_ms_band[:, 0:1,:]), lr_ms_band), dim=1)
    # Bx(C+1)xrefC
    sol = torch.linalg.lstsq(lr_ms_band.transpose(1, 2), down_hr_pan.transpose(1, 2))

    sol = sol.solution.transpose(1, 2)
    return sol 