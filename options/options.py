# from distutils.command.config import config
import os
from collections import OrderedDict
from datetime import datetime
import yaml
# import torch
# from yaml.events import NodeEvent
import data.fileio as fileio
import shutil
import argparse

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

yml_Loader, yml_Dumper = OrderedYaml()


def _common_args():
    # basic args
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-gpuid', type=str, default='0,', help='Define the gpu id to run code.')
    parser.add_argument('-net_arch', type=str, default=None, help='The network to run.')
    parser.add_argument('-pretrained_path', type=str, default=None)
    parser.add_argument('-scale', type=float, default=None, help='The upscale for the running network.')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    ################  { logger }  ################
    parser.add_argument('-tag', type=str, default=None)
    parser.add_argument('-exp_key', type=str, default=None)
    ################  { dataset }  ################
    parser.add_argument('-setmode', type=str, default=None)
    parser.add_argument('-settype', type=str, default=None)
    parser.add_argument('-is_srf', type=str, default=None)
    parser.add_argument('-is_degrade', type=str, default=None)
    parser.add_argument('-get_MTF', type=str, default=None)
    parser.add_argument('-choose_band', type=str, default=None)
    parser.add_argument('-scale_change', type=str, default=None)
    parser.add_argument('-scale_delta', type=float, default=None)
    parser.add_argument('-low_thre', type=float, default=None)
    parser.add_argument('-high_thre', type=float, default=None)

    ########## network setting  ##########
    parser.add_argument('-convDim', type=int, default=None)
    parser.add_argument('-numHeads', type=int, default=None)
    parser.add_argument('-patchSize', type=int, default=None)
    parser.add_argument('-poolSize', type=int, default=None)
    parser.add_argument('-numLayers', type=int, default=None)
    parser.add_argument('-use_dynamic', type=str, default=None)
    parser.add_argument('-int_type', type=str, default=None)
    parser.add_argument('-temper_step', type=float, default=None)
    parser.add_argument('-num_res', type=int, default=None)
    parser.add_argument('-num_cycle', type=int, default=None)
    parser.add_argument('-norm_type', type=str, default=None)
    parser.add_argument('-scope', type=str, default=None)

    return parser

def add_train_args():
    parser = _common_args()

    # for logging
    parser.add_argument('-log_dir', type=str, default=None, help='The path of saved model.')
    parser.add_argument('-tags', type=str, nargs='+', default=None)
    
    ########### dataset-setting ##########
    parser.add_argument('-repeat', type=int, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-patch_size', type=int, default=None)

    ########### optimizer-setting ##########
    parser.add_argument('-loss_name', type=str, default=None)
    parser.add_argument('-optimType', type=str, default=None)
    parser.add_argument('-learning_rate', type=float, default=None)
    parser.add_argument('-lr_scheme', type=str, default=None)
    parser.add_argument('-warmUpEpoch', type=int, default=None)
    parser.add_argument('-lrStepSize', type=int, default=None)
    parser.add_argument('-acuSteps', type=int, default=None)
    parser.add_argument('-num_epochs', type=float, default=None)

    ################  { training setting }  ################
    parser.add_argument('-mask_training', type=str, default=None)
    parser.add_argument('-pretrain', type=str, default=None)
    
    return parser

def add_test_args():
    parser = _common_args()
    parser.add_argument('-results_dir', type=str, default=None)
    parser.add_argument('-add_log', type=str, default=None)
    return parser


def parse(args):
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yml_Loader)
    
    def add_args2yml(chars, optdic):
        if getattr(args, chars, None) is not None:
            optdic[chars] = getattr(args, chars)
        return optdic
    
    opt = add_args2yml('exp_key', opt)
    
    if opt['mode']=='train':
        opt['solver'] = add_args2yml('pretrain', opt['solver'])
        opt['solver'] = add_args2yml('pretrained_path', opt['solver'])
        if opt['solver']['pretrain'] == 'resume':
            opt = opt['solver']['pretrained_path']
            assert os.path.isfile(opt), \
                'The models of %s does not exist.'%opt
            opt = os.path.join(os.path.dirname(opt), 'options.yml') 
            with open(opt, mode='r') as f:
                opt = yaml.load(f, Loader=yml_Loader)
            return opt
    else:
        opt = add_args2yml('tag', opt)
        if_add_log = getattr(args, 'add_log', None)
        if if_add_log and if_add_log.lower()=='true': 
            opt['add_log'] = True


    ################  { data Setting }  ################
    run_range = opt['run_range']
    for dataname, dataset in opt['datasets'].items():
        phase, _ = dataname.split('_')
        dataset['phase']=phase
        dataset['name'] = dataname
        dataset['run_range'] = run_range
        # dataset['choose_band'] = (opt['networks']['net_arch'].upper()=='BIEDN' and 'train' in phase)
        dataset = add_args2yml('setmode', dataset)
        dataset = add_args2yml('settype', dataset)
        if getattr(args, 'get_MTF') is not None:
                dataset['get_MTF'] = (getattr(args, 'get_MTF').lower() == 'true')
        if getattr(args, 'is_srf') is not None:
                dataset['is_srf'] = (getattr(args, 'is_srf').lower() == 'true')
        if getattr(args, 'is_degrade') is not None:
            dataset['is_degrade'] = (getattr(args, 'is_degrade').lower() == 'true')
        if getattr(args, 'choose_band') is not None:
                dataset['choose_band'] = (getattr(args, 'choose_band').lower() == 'true')
        if getattr(args, 'scale_change') is not None:
                dataset['scale_change'] = (getattr(args, 'scale_change').lower() == 'true')
        # dataset['scaledict']={'LR':1, 'REF': opt['scale'], 'GT':opt['scale']}
        if 'train' == phase:
            dataset = add_args2yml('scale_delta', dataset)
            dataset = add_args2yml('low_thre', dataset)
            dataset = add_args2yml('high_thre', dataset)
            dataset = add_args2yml('repeat', dataset)
            dataset = add_args2yml('batch_size', dataset)
            dataset = add_args2yml('patch_ize', dataset)
        opt['datasets'][dataname]=dataset

        ################  { network Setting }  ################
        opt['networks']['scale'] = opt['scale']
        opt['networks'] = add_args2yml('net_arch', opt['networks'])
        opt['networks'] = add_args2yml('convDim', opt['networks'])
        opt['networks'] = add_args2yml('numHeads', opt['networks'])
        opt['networks'] = add_args2yml('numLayers', opt['networks'])
        opt['networks'] = add_args2yml('patchSize', opt['networks'])
        opt['networks'] = add_args2yml('poolSize', opt['networks'])
        opt['networks'] = add_args2yml('use_dynamic', opt['networks'])
        opt['networks'] = add_args2yml('int_type', opt['networks'])
        opt['networks'] = add_args2yml('temper_step', opt['networks'])
        opt['networks'] = add_args2yml('num_cycle', opt['networks'])
        opt['networks'] = add_args2yml('num_res', opt['networks'])
        opt['networks'] = add_args2yml('norm_type', opt['networks'])
        opt['networks'] = add_args2yml('scope', opt['networks'])    
        opt['networks']['LRdim'] = opt['datasets'][dataname]['LRdim']
        opt['networks']['REFdim'] = opt['datasets'][dataname]['REFdim']
    
    
    ################  { optim&lr_rate Setting }  ################
    opt['solver'] = add_args2yml('pretrain', opt['solver'])
    opt['solver'] = add_args2yml('pretrained_path', opt['solver'])
    if opt['mode'] == 'train':
        opt = add_args2yml('mask_training', opt)
        opt['solver'] = add_args2yml('loss_name', opt['solver'])
        opt['solver'] = add_args2yml('optimType', opt['solver'])
        opt['solver'] = add_args2yml('learning_rate', opt['solver'])
        opt['solver'] = add_args2yml('lr_scheme', opt['solver'])
        opt['solver'] = add_args2yml('warmUpEpoch', opt['solver'])
        opt['solver'] = add_args2yml('lrStepSize', opt['solver'])
        opt['solver'] = add_args2yml('acuSteps', opt['solver'])
        opt['solver'] = add_args2yml('num_epochs', opt['solver'])
    
    ################  { Logging setting }  ################
    opt['timestamp'] = get_timestamp() # logging date and time
    if opt['mode']=='train': # train
        opt['logger'] = add_args2yml('tag', opt['logger'])
        opt['logger'] = add_args2yml('tags', opt['logger'])
        config_str = '%s' %(opt['networks']['net_arch'])
        if opt['logger'].get('tag', None) is not None: config_str = config_str + '_' + opt['logger']['tag']
        config_str = getattr(args, 'log_dir', '') + config_str
        opt = set_logger_dir(opt, config_str)   
    else: # test
        opt = add_args2yml('results_dir', opt)
        opt['results_dir'] = os.path.join( opt.get('results_dir', ''), opt['networks']['net_arch']) 
    opt = dict_to_nonedict(opt)
    return opt


def set_logger_dir(opt, config_str):
    if opt['solver']['pretrain'] == 'finetune': # finetune
        assert os.path.isfile(opt['solver']['pretrained_path']), \
            'The models of %s does not exist.'%opt
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        exp_path += '_finetune'
    else:
        exp_path = os.path.join('experiments', config_str)
        
    exp_path = os.path.relpath(exp_path)
    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['logger'].update(path_opt)
    return opt

def creat_logger_dir(opt, Dumper=yml_Dumper):
    fileio.mkdir_and_rename(opt['logger']['exp_root'])  # rename old experiments if exists
    fileio.mkdir_and_rename(opt['logger']['epochs'])  # rename old experiments if exists
    fileio.mkdir_and_rename(opt['logger']['records'])  # rename old experiments if exists
    save_setting(opt, Dumper)
    print("===> Experimental DIR: [%s]" % opt['logger']['exp_root'])
    pass

def save_setting(opt, Dumper):
    dump_dir = opt['logger']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.yml')
    network_file = opt["networks"]['net_arch'] + '.py'
    shutil.copy('./networks/' + network_file, os.path.join(dump_dir, network_file))
    with open(dump_path, 'w') as dump_file:
        yaml.dump(nonedict_to_dict(opt), dump_file, Dumper=Dumper)


class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
    
def nonedict_to_dict(opt):
    if isinstance(opt, NoneDict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = nonedict_to_dict(sub_opt)
        return new_opt
    elif isinstance(opt, list):
        return [nonedict_to_dict(sub_opt) for sub_opt in opt]
    else:
        return opt


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')