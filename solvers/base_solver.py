import torch
import torch.nn as nn

class BaseSolver(object):
    def __init__(self, opt):
        

        # log and vis scheme
        


    def feed_data(self, batch):
        pass

    def train_step(self):
        pass

    def test(self):
        pass

    def _forward_x8(self, x, forward_function):
        pass

    def _overlap_crop_forward(self, upscale):
        pass

    def get_current_log(self):
        pass

    def get_current_visual(self):
        pass

    def get_current_learning_rate(self):
        pass

    def set_current_log(self, log):
        pass

    def update_learning_rate(self, epoch):
        pass

    def save_checkpoint(self, epoch, is_best):
        pass

    def load(self):
        pass

    def save_current_visual(self, epoch, iter):
        pass

    def save_current_log(self):
        pass

    def print_network(self):
        pass

    
