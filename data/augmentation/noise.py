import numpy as np


def np_add_noise(x, noise_type, noise_value):
    # if noise != '.':
    if np.random.random()>0.5:
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)
        else:
            raise TypeError('Cannot recognize this [%s] type of noise for augmentation'%noise_type)
        x_noise = x + noises
            # x_noise = x.astype(np.int16) + noises.astype(np.int16)
            # x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x