import numpy as np
import glob
import os
from utils.util import pan_calc_metrics_all
import pandas as pd
import csv
import tifffile


if __name__ == '__main__':
    
    dataset_list = ['PAirMax/GE_Tren_Urb', 'PAirMax/GE_Lond_Urb']
    img_range_list = [2**11-1]*len(dataset_list)
    sensor_list = ['GE', 'GE']
    methods_list = ['EXP', 'GSA', 'MTF-GLP', 'BDSD-PC', 'PRACS', 'MTF-GLP-FS']
    for idx_dataset in range(len(dataset_list)):
        img_range = img_range_list[idx_dataset]
        dataset = dataset_list[idx_dataset]
        for idx_method in range(len(methods_list)):
            method = methods_list[idx_method]
            sr_dir = '../PanSharp_dataset/%s/result_FR/%s/'%(dataset, method)
            # gt_dir = '../PanSharp_dataset/%s/test/GT/'%dataset
            # ref_dir = '../PanSharp_dataset/%s/test/REF_FR/'%dataset
            gt_dir = '../PanSharp_dataset/%s/FR/MS_LR.tif'%dataset
            ref_dir = '../PanSharp_dataset/%s/FR/PAN.tif'%dataset
            
            sr_files  = glob.glob(os.path.join(sr_dir, '*.npy'))
            metrics = None
            for idx_img, path_img in enumerate(sr_files):
                sr = np.load(path_img).astype(np.float32)
                name_sr = os.path.basename(path_img).replace('SR', '')
                # gt = np.load(os.path.join(gt_dir, name_sr)).astype(np.float32)
                # ref = np.load(os.path.join(ref_dir, name_sr)).astype(np.float32)
                # gt = imageio.imread(gt_dir).astype(np.float32)
                # ref = imageio.imread(ref_dir).astype(np.float32)
                gt = tifffile.imread(gt_dir).astype(np.float32)
                ref = tifffile.imread(ref_dir).astype(np.float32)
                tmp_metrics = pan_calc_metrics_all({'OUT':sr, 'LR':gt, 'REF':ref, 'SENSOR':sensor_list[idx_dataset]}, 4, img_range, True)
                if metrics is None: metrics = {key: [] for key in tmp_metrics.keys()}
                for key, value in tmp_metrics.items(): metrics[key].append(value)
                # print('%s: %s'%(method, tmp_metrics))
            ave_metrics = {key:sum(value)/len(value) for key, value in metrics.items()}
            print('%s |%s | Average :%s'%(dataset, method, ave_metrics))
            ave_metrics['methods']=method
            ave_metrics['datasets']=dataset
            with open('fr_results.csv', 'a') as f:
                w = csv.DictWriter(f, ave_metrics.keys())
                w.writeheader()
                w.writerow(ave_metrics)
            
                
    # print(ave_metrics)
        '../PanSharp_dataset/Airbus_PSdataset/P1A/test/GT/IMG_PHR1A_MS_201909161238065_SEN_5589502101-2_R1C1.npy'
        