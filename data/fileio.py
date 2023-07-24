import os 
import glob
from datetime import datetime
import numpy as np
import imageio

from data.trans_data import multidata


IMG_EXT = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
           '.bmp', '.BMP']


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

################  { dir operation }  ################
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def get_image_paths(path, ext):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = glob.glob(os.path.join(path, '*'+ext))
    images.sort()
    assert images, '[%s] has no valid file' % path
    return images

################  { io operation }  ################
@multidata
def read_img(path, ext):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if ext == '.npy':
        img = np.load(path)
    elif ext in IMG_EXT:
        import imageio
        img = imageio.imread(path, pilmode='RGB')
    elif ext == '.mat':
        from scipy import io as sciio
        img = sciio.loadmat(file_name=path)
    elif ext in ('.tif', '.TIF', '.tiff', '.TIFF'):
        from skimage import io as skimgio
        img = skimgio.imread(path)
    else:
        raise NotImplementedError(
            'Cannot read this type (%s) of data' % ext)
    if isinstance(img, np.ndarray) and img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

def save_img(path, img, ext):
    if ext == '.npy':
        np.save(path+ext, img)
        print('saving npy img to %s'%(path+ext))
    elif ext in IMG_EXT:
        import imageio
        imageio.imwrite(path+ext, img, ext)
    elif ext == '.mat':
        from scipy import io as sciio
        sciio.savemat(path+ext,{'data':img})
    elif ext in ('tif', 'TIF', 'tiff', 'TIFF'):
        import skimage.external.tifffile as skimg_tiff
        skimg_tiff.imsave(path+ext)
    else:
        raise NotImplementedError(
            'Cannot read this type (%s)/(%s) of data' %(ext, type(img)))