﻿
from copy import deepcopy
import os
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.optim as optim
from networks import create_model, init_weights
# from networks import loss as myloss


class SRSolver():
    def __init__(self, opt):
        super(SRSolver, self).__init__()
        self.opt = opt
        self.is_train = (opt['mode']=='train')
        self.scale = opt['scale']
        # self.use_chop = opt['use_chop']
        # self.self_ensemble = opt['self_ensemble']
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        self._set_model()
        print('===> Solver Initialized : [%s] || Use GPU : [%s]'% (
            self.__class__.__name__, self.use_gpu))
            
    def _set_model(self,):
        if not self.is_train: # test
            self.model = create_model(self.opt['networks'])
        else: # train
            self.exp_root = self.opt['logger']['exp_root']
            self.checkpoint_dir = self.opt['logger']['epochs']
            self.records_dir = self.opt['logger']['records']
            self.save_ckp_step = self.opt['solver']['save_ckp_step']
            self.log = {
                'epoch': 1,
                'best_pred' : None,
                'best_epoch': 1,
                'records' : { 
                        'train_loss': [],
                        'val_loss': [],
                        'lr': []}
            }
            self.solver_opt = self.opt['solver']
            self.model = create_model(self.opt['networks'])
            self._set_loss()
            self._set_optimizer()
            self._set_lr_schedulre()
        self.init_weight()
    
    def _set_loss(self,):
        if self.solver_opt['loss_name'].lower() == 'selfloss':
            self.loss = self.model.loss
        else:
            self.loss = myloss.make(self.solver_opt['loss_name'], self.solver_opt)
        
    def _set_optimizer(self,):
        weight_decay = self.solver_opt['weight_decay'] if self.solver_opt['weight_decay'] else 0
        optim_type = self.solver_opt['optimType'].upper()
        if optim_type == 'MYOPTIM':
            self.optimizer = self.model.optimizer()
        elif optim_type == "ADAM":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.solver_opt['learning_rate'], weight_decay=weight_decay)
        else:
            raise NotImplementedError(
                'Loss type [%s] is not implemented!' % optim_type)
        print("optimizer: ", self.optimizer)

    def _set_lr_schedulre(self,):
        if self.solver_opt['lr_scheme'].lower() == 'constant':
            self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer)
        elif self.solver_opt['lr_scheme'].lower() == 'warm_up':
            warmUpEpoch = self.solver_opt['warmUpEpoch']
            lrStepSize = self.solver_opt['lrStepSize']
            lrLambda = lambda epoch: epoch/warmUpEpoch if epoch < warmUpEpoch \
                else 0.5**(epoch//lrStepSize)
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lrLambda)
        elif self.solver_opt['lr_scheme'].lower() == 'cosanneal':
            Tmax = 32  # or 64
            lrmax = self.solver_opt['learning_rate']
            lrmin = 0.000001
            warmUpEpoch = self.solver_opt['warmUpEpoch']
            lrStepSize = self.solver_opt['lrStepSize']
            lrLambda = lambda epoch: epoch / warmUpEpoch if epoch < warmUpEpoch else \
                (lrmin+0.5*(lrmax-lrmin)*(1.0+math.cos((epoch-warmUpEpoch)/Tmax*math.pi)))
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lrLambda)
        elif self.solver_opt['lr_scheme'].lower() == 'multisteplr':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            [200, 400, 600, 800], 0.5)
        else:
            raise NotImplementedError('Only MultiStepLR scheme is supported!')
        print("lr_scheduler: %s " % (self.solver_opt['lr_scheme']))

    def set_current_log(self, log):
        self.log = log
        
    def get_current_log(self,):
        return self.log

    def save_checkpoint(self, is_best):
        """
        save checkpoint to experimental dir
        """ # TODO: checkpoint_dir
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth') #
        ckp = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'log':self.log
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace(
                'last_ckp', 'best_ckp'))
            torch.save(ckp, filename.replace('last_ckp', 'best_ckp'))

    def _overlap_crop_forward(self, imgdict:dict, scaledict:dict, shave=10, min_size=100000):
        """
        chop for less memory consumption during test
        imgdict: a dict constructed by the network input, the resoution of the item increase with the list index
        scaledict: a tuple of scale ratio between each item and the first one.
        """
        n_GPUs = 2
        scale = self.scale
        b, c, h, w = imgdict['LR'].size()
        # print('h: %f, w:%f'%(h, w))
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        
        sizedict = {t_key:(h_size*t_scale, w_size*t_scale) for t_key, t_scale in scaledict.items()}
        mtf = imgdict.pop('MTF', 'null')
        imgtuple_11 = {t_key:t_img[:,:,:sizedict[t_key][0],:sizedict[t_key][1]] for t_key, t_img in imgdict.items() }
        imgtuple_12 = {t_key:t_img[:,:,:sizedict[t_key][0],-sizedict[t_key][1]:] for t_key, t_img in imgdict.items()}
        imgtuple_21 = {t_key:t_img[:,:,-sizedict[t_key][0]:,:sizedict[t_key][1]] for t_key, t_img in imgdict.items()}
        imgtuple_22 = {t_key:t_img[:,:,-sizedict[t_key][0]:,-sizedict[t_key][1]:] for t_key, t_img in imgdict.items()}
        # imgtuple_12 = [ t_img[:,:,:t_hsize,-t_wsize:] for t_img, (t_hsize, t_wsize) in zip(imgtuple, size_list)]
        # imgtuple_21 = [ t_img[:,:,-t_hsize:,:t_wsize] for t_img, (t_hsize, t_wsize) in zip(imgtuple, size_list)]
        # imgtuple_22 = [ t_img[:,:,-t_hsize:,-t_wsize:] for t_img, (t_hsize, t_wsize) in zip(imgtuple, size_list)]
        
        imgdict_list = [imgtuple_11, imgtuple_12, imgtuple_21, imgtuple_22]
        
        if w_size * h_size <= min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                imgdict_batch = imgdict_list[i]
                for j in range(1, n_GPUs):
                    imgdict_batch = {t_key: torch.cat((t_value, imgdict_list[i+j][t_key]), dim=0) 
                                    for t_key, t_value in imgdict_batch.items() 
                                    if t_key !='MTF'
                    }
                if mtf is not 'null':imgdict_batch['MTF'] = mtf.repeat(n_GPUs, 1, 1, 1)
                sr_batch_temp = self.model(imgdict_batch)
                sr_batch = sr_batch_temp
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(t_imgdict, scaledict, shave=shave, min_size=min_size) \
                for t_imgdict in imgdict_list
                ]

        h, w = scale * h, scale * w
        h1_half, w1_half = scale * h_half, scale * w_half
        h2_half, w2_half = h-h1_half, w-w1_half
        # h_size, w_size = scale * h_size, scale * w_size
        # shave *= scale
        output = imgdict['LR'].new(b, c, h, w)
        output[:, :, :h1_half, :w1_half] \
            = sr_list[0][:, :, :h1_half, :w1_half]
        output[:, :, :h1_half, w1_half:] \
            = sr_list[1][:, :, :h1_half, -w2_half:]
        output[:, :, h1_half:, :w1_half] \
            = sr_list[2][:, :, -h2_half:, :w1_half]
        output[:, :, h1_half:, w1_half:] \
            = sr_list[3][:, :, -h2_half:, -w2_half:]
        return output

    def init_weight(self):
        """
        load or initialize network
        """
        if not self.is_train or self.opt['solver']['pretrain'] is not None:
            model_path = self.opt['solver']['pretrained_path']
            print('===> Loading model from [%s]...' % model_path)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['state_dict']) #for test and finetune
            if self.opt['solver']['pretrain'] == 'resume':
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.log = checkpoint['log']
                self.log['epoch'] += 1
        else:
            pass
            # self.model.weight_init()
                
    def load_func(self, checkpoint):
        new_ckp = {}
        for name, name_weight in checkpoint.items():
            if name in self.model.state_dict().keys():
                new_ckp[name] = name_weight
        self.model.load_state_dict(new_ckp)
        
    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self,):
        if self.opt['lr_scheme'] == 'net_lr_scheme':
            self.model.update_learning_rate()
        else:
            self.scheduler.step()
    
    def save_current_records(self):
        records = deepcopy(self.log['records'])
        for key in records.keys():
            records[key].append(records[key][self.log['best_epoch'] - 1])
        res_index = list(range(1, self.log['epoch'] + 1))
        res_index.append('Best epoch' + str(self.log['best_epoch']))
        data_frame = pd.DataFrame(
            data={key: value for key, value in records.items()},
            index=res_index
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                          index_label='epoch')

    
    def cal_flops(self,):
        # from ptflops import get_model_complexity_info
        # def input_constructor(input_size_tuple):
        #     return  {'batchData':{'LR' :torch.rand((1, 31, 18, 18), device=next(self.model.parameters()).device),
        #                           'REF':torch.rand((1, 3, 72, 72), device=next(self.model.parameters()).device)}}
        # with torch.cuda.device(0):
        #     macs, params = get_model_complexity_info(self.model, (31, 64, 64), 
        #                                              input_constructor=input_constructor, 
        #                                              as_strings=False, verbose=True)
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        from thop import profile, clever_format
        batchData = {'LR' :torch.rand((1, 31, 18, 18), device=next(self.model.parameters()).device),
                    'REF':torch.rand((1, 3, 72, 72), device=next(self.model.parameters()).device)}
        macs, params = profile(self.model, inputs=(batchData, )) # custom_ops={YourModule: count_your_model})
        macs, params = clever_format([macs, params], "%.3f")
        print('Macs: %s, #Params: %s'%(macs, params))