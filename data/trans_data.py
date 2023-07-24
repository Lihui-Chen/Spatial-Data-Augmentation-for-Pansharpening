import numpy as np
import torch

# def np2Tensor(img, img_range, run_range=1):
#     def _np2Tensor(img):
#         np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))/1.0
#         tensor = torch.from_numpy(np_transpose).float()
#         tensor.mul_(run_range / img_range)
#         return tensor
#     if isinstance(img, dict):
#         return {t_key: _np2Tensor(tensor) for t_key, tensor in img.items()}
#     elif isinstance(img, np.ndarray):
#         return _np2Tensor(img)
#     elif isinstance(img, (list, tuple)):
#         return [_np2Tensor(tmp) for tmp in img]
#     else:
        # raise(TypeError('Cannot transfer %s of np.ndarray to tensor'%type(img)))

# def tensor2np(img, img_range, run_range=1, is_quantize=False):
#     def _Tensor2numpy(tensor):
#         array = np.transpose(
#             map2img_range(tensor, run_range, img_range, is_quantize).numpy(), (1, 2, 0)
#         )
#         return array
#     if isinstance(img, dict):
#         return {t_key: _Tensor2numpy(tensor) for t_key, tensor in img.items()}
#     elif isinstance(img, torch.Tensor):
#         return _Tensor2numpy(img)
#     elif isinstance(img, (list, tuple)):
#         return [_Tensor2numpy(tmp) for tmp in img]
#     else:
#         raise(TypeError('Cannot transfer %s of tensor to numpy'%type(img)))


def multidata(func):
    def inner(input, *args, **kvargs):
        if isinstance(input, dict):
            return {t_key: func(tensor, *args, **kvargs) for t_key, tensor in input.items()}
        elif isinstance(input, (list, tuple)):
            return [func(tmp, *args, **kvargs) for tmp in input]
        else:
            return func(input, *args, **kvargs)
    return inner

@multidata
def np2tensor(img, img_range, run_range=1):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))/1.0
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(run_range / img_range)
    return tensor

@multidata
def tensor2np(tensor, img_range, run_range=1, is_quantize=False):
    array = np.transpose(
        map2img_range(tensor, img_range, run_range, is_quantize).numpy(), (1, 2, 0)
    )
    return array

def map2img_range(img:torch.Tensor, img_range:float, run_range=1, is_quantize=False):
    if is_quantize:
        return img.mul(img_range / run_range).clamp(0, int(img_range)).round()
    else:
        return img.mul(img_range / run_range).clamp(0, img_range)
    
@multidata
def data2device(batch:torch.Tensor, device):
    return batch.to(device)