import time
import os
import options.options as option
from solvers import SRSolver
from data import create_dataloader
from data import create_dataset
import numpy as np
from data import fileio, trans_data
import torch
from utils.util import pan_calc_metrics_all as all_metrics
from utils.util import pan_calc_metrics_rr as rr_metrics
from utils.vistool import hist_line_stretch
import cv2

@trans_data.multidata
def data2device(batch:torch.Tensor, device):
    return batch.to(device)

select_band = lambda x: (2, 1, 0) if x.shape[2]>2 else 0
choose_key = lambda x: x.keys[0]

def dictbatch2batchdict(dictbatch):
    batchdict = []
    key = list(dictbatch.keys())[0]
    for i in range(dictbatch.get(key).shape[0]):
        batchdict.append({key: value[i] for key, value in dictbatch.items()})
    return batchdict

def validate(net, loss, dataloader, dataopt, is_visualize_out=False, is_visGT=False):
    run_range = dataopt['run_range']
    img_range = dataopt['img_range']
    scale = dataopt['scaledict']['REF']
    dataname = dataopt['name']
    valLossTime = {'val_loss': 0, 'time':0}
    isFR = (dataopt['settype'] == 'FR')
    net.eval()
    mask=None
    test_metrics = None
    vis_tbloggin = {}
    for batchIdx, (dataBatch, mask) in enumerate(dataloader):
        with torch.no_grad():
            # GT = dataBatch.get('GT', dataBatch.get('HR', None))
            ori_databatch = dataBatch
            dataBatch = data2device(dataBatch, next(net.parameters()).device)
            strat_time = time.time()
            out = net(dataBatch, mask)
            end_time = time.time()
            valLossTime['time'] += (end_time-strat_time)

            # valLossTime['val_loss'] += loss(out, dataBatch).item()*out.shape[0]
            tmploss = loss(out, dataBatch)
            if isinstance(tmploss, (list, tuple)):
                tmploss, grad_gt, grad_out = tmploss
            if isinstance(out, (list, tuple)): out = out[0]
            valLossTime['val_loss'] += tmploss.item()*out.shape[0]
            # valLossTime['val_loss'] += tmploss.item()*len(out)
            
        # out = out.split(1, dim=0) if isinstance(out, torch.Tensor) else [tmp.cpu() for tmp in out]
        # out = trans_data.tensor2np([tmp.squeeze(dim=0).cpu() for tmp in out], img_range, run_range, is_quantize=(img_range>=255))
        out = out.cpu() 
        ori_databatch['OUT'] = out
        ori_databatch = dictbatch2batchdict(ori_databatch)
        for idx, tmp_dict in enumerate(ori_databatch):
            mtf = tmp_dict.pop('MTF') if tmp_dict.get('MTF') is not None else None
            tmp_dict = trans_data.tensor2np(tmp_dict, img_range, run_range, is_quantize=(img_range>=255))
            tmp_dict['SENSOR'] = dataname.split('_')[-1]
            if mtf is not None: mtf = trans_data.tensor2np(mtf.squeeze(dim=0), 1, 1, is_quantize=False)
            tmp_dict['MTF'] = mtf
            tmpMetrics = all_metrics(tmp_dict, scale=scale, img_range=img_range, FR=isFR)
            if test_metrics is None:
                test_metrics = {key: value for key,value in tmpMetrics.items()}
            else:
                test_metrics={key:test_metrics[key]+value
                              for key, value in tmpMetrics.items()}
            ori_databatch[idx] = tmp_dict
                     
        if is_visGT and batchIdx==0:
            if vis_tbloggin.get('GT') is None: vis_tbloggin['GT'] = []
            GT = [tmpdict['GT'][:,:,select_band(tmpdict['GT'])] for tmpdict in ori_databatch]
            GT = [hist_line_stretch(tmp.astype(np.float), nbins=255) for tmp in GT]
            GT = np.concatenate(GT, axis=1)
            vis_tbloggin['GT'] = GT

            # if vis_tbloggin.get('grad_gt') is None: vis_tbloggin['grad_gt'] = []
            # grad_gt = grad_gt.split(1, dim=0)
            # grad_gt = trans_data.tensor2np([tmp.squeeze(dim=0).cpu() for tmp in grad_gt], img_range, run_range, is_quantize=(img_range>=255))
            # grad_gt = [tmp[:,:,select_band(tmp)] for tmp in grad_gt]
            # grad_gt = [hist_line_stretch(tmp.astype(np.float), nbins=255) for tmp in grad_gt]
            # grad_gt = np.concatenate(grad_gt, axis=1)
            # vis_tbloggin['grad_gt'].append(grad_gt)
            # vis_tbloggin['GT'] = GT
            # GT = GT[:,:,(2,1,0)]
            # GT = [hist_line_stretch(tmp.astype(np.float), nbins=255) for tmp in GT]
            # vis_tbloggin['GT'].append(GT)
            
        if is_visualize_out and batchIdx==0: 
            if vis_tbloggin.get('OUT') is None: vis_tbloggin['OUT'] = [] 
            out = [tmpdict['OUT'][:,:,select_band(tmpdict['OUT'])] for tmpdict in ori_databatch]
            out = [hist_line_stretch(tmp.astype(np.float), nbins=255) for tmp in out]
            out = np.concatenate(out, axis=1)
            vis_tbloggin['OUT']=out

            # if vis_tbloggin.get('grad_out') is None: vis_tbloggin['grad_out'] = [] 
            # grad_out = grad_out.split(1, dim=0)
            # grad_out = trans_data.tensor2np([tmp.squeeze(dim=0).cpu() for tmp in grad_out], img_range, run_range, is_quantize=(img_range>=255))
            # grad_out = [tmp[:,:,select_band(tmp)] for tmp in grad_out]
            # grad_out = [hist_line_stretch(tmp.astype(np.float), nbins=255) for tmp in grad_out]
            # if len(grad_out)>0:
            #     grad_out = np.concatenate(grad_out, axis=1)
            #     vis_tbloggin['grad_out'].append(grad_out)
            # out = out[:,:,(2,1,0)] 
            # out = [hist_line_stretch(tmp.astype(np.float), nbins=255) for tmp in out]
            # vis_tbloggin['Out'].append(out)
        
    test_metrics = {**test_metrics, **valLossTime}
    test_metrics = {tkey: tvalue/len(dataloader.dataset) for tkey, tvalue in test_metrics.items()}
    # vis_tbloggin = {key:np.concatenate(value, axis=1) for key, value in vis_tbloggin.items()}
    return test_metrics, vis_tbloggin


def test(solver:SRSolver, net, dataloader, dataopt, is_saveImg=True, savedir= None, is_visualize_out=False, is_visualize_gt=False):
    run_range = dataopt['run_range']
    img_range = dataopt['img_range']
    scale = dataopt['scaledict']['REF']
    dataname = dataopt['name']
    isFR = (dataopt['settype'] == 'FR') or ('FR' in dataopt['name'])
    valTime = []
    vis_rlt = dict()
    net.eval()
    test_metrics = None
    for imgIdx, (dataBatch, batchPath) in enumerate(dataloader):
        with torch.no_grad():
            strat_time = time.time()
            ori_databatch = dataBatch
            print(batchPath['REF'])
            dataBatch = data2device(dataBatch, next(net.parameters()).device)
            # GT = dataBatch.get('GT', dataBatch.get('HR', None))
            if dataopt['use_chop'] and 'FR' in dataname:
                out = solver._overlap_crop_forward(dataBatch, dataopt['scaledict'])
            else:
                out = net(dataBatch)
            
            # loss = 
            end_time = time.time()
            valTime.append(end_time-strat_time)
        out = trans_data.tensor2np(out.squeeze().cpu(), img_range, run_range, is_quantize=(img_range>=255))
        mtf = ori_databatch.pop('MTF') if ori_databatch.get('MTF') is not None else None
        ori_databatch = {key:value.squeeze(dim=0) for key, value in ori_databatch.items()}
        ori_databatch = trans_data.tensor2np(ori_databatch, img_range, run_range, is_quantize=(img_range>=255))
        ori_databatch['OUT'] = out
        ori_databatch['SENSOR'] = dataname.split('_')[-1]
        if mtf is not None: mtf = trans_data.tensor2np(mtf.squeeze(dim=0), 1, 1, is_quantize=False)
        ori_databatch['MTF'] = mtf
        
        tmpMetric = all_metrics(ori_databatch, scale=scale, img_range=img_range, FR=isFR)
        if imgIdx==0:
            test_metrics = {tkey:[tvalue,] for tkey, tvalue in tmpMetric.items()}
        else:
            for tkey, tvalue in tmpMetric.items(): test_metrics[tkey].append(tvalue)
        print('processing the %d-th image'%(imgIdx+1))
        if is_saveImg: 
            path = os.path.join('results', dataname, savedir)
            if imgIdx == 0: print(path)
            print('saving the %d-th image'%(imgIdx+1))
            if not os.path.isdir(path): os.makedirs(path)
            path = os.path.join(path, os.path.basename(batchPath['LR'][0])[:-4])
            fileio.save_img(path, out, '.npy')
     
    test_metrics = {**test_metrics, **{'Time':valTime}}
    # test_metrics = {**{'Time':valTime}}
    test_metrics = {tkey: np.mean(np.array(tvalue)) for tkey, tvalue in test_metrics.items()}
    return test_metrics

# def target_adaptive_test(dataloader, finetune_tot_iters):
#     for inp_idx, (batch_dat, batch_pth) in enumerate(dataloader):
#         for iter_idx in range(finetune_tot_iters):
#             batch
            

def main():
    args = option.add_test_args().parse_args()
    opt = option.parse(args)
    opt = option.dict_to_nonedict(opt)
    
    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['net_arch'].upper()
    # create test dataloader
    bm_names = []
    test_loaders = {}
    for dataname, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset([dataset_opt])[0]
        test_loader = create_dataloader(test_set, 1, shuffle=False, num_workers=1)
        # test_loaders.append(test_loader)
        test_loaders[dataname] = test_loader
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (
            dataname, len(test_set)))

    # create solver (and load model)
    solver = SRSolver(opt)
    print("==================== Start Test========================")
    print("Method: %s || Scale: %d || Degradation: %s" %(model_name, scale, degrad))
    for dataname, dataloader in test_loaders.items():
        testMetrics = test(solver, solver.model, dataloader, opt['datasets'][dataname], is_saveImg=True, savedir=opt['results_dir'])
        # test(solver.model, dataloader, opt, dataname, is_saveImg=True, savedir=opt['results_dir'])
        print('======= The results on %s are as following. ======='%dataname)
        print("| Method |%s\n| ------ %s|\n| %s | %s |"
                % (''.join([' {:7}|'.format(key) for key in testMetrics.keys()]),
                    '| ------ '*len(testMetrics.keys()), network_opt['net_arch'].upper()[:6],
                    ' | '.join(['%.4f'%value for value in testMetrics.values()]))
            )
    print("======================= END =======================")


if __name__ == '__main__':
    main()
