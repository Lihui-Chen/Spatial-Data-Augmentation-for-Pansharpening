import numpy as np
import glob
import csv
import os
import imageio
from utils.util import pan_calc_metrics_all
if __name__ == '__main__':
    dataset_list = ['PAirMax/GE_Tren_Urb', 'PAirMax/GE_Lond_Urb']
    img_range_list = [2**11-1]*len(dataset_list)
    methods_list = ['EXP', 'GSA', 'MTF-GLP', 'BDSD-PC', 'PRACS', 'MTF-GLP-FS']
    for idx_dataset in range(len(dataset_list)):
        img_range = img_range_list[idx_dataset]
        dataset = dataset_list[idx_dataset]
        for idx_method in range(len(methods_list)):
            method = methods_list[idx_method]
            sr_dir = '../PanSharp_dataset/%s/result/%s/'%(dataset, method)
            gt_dir = '../PanSharp_dataset/%s/test/GT/'%dataset
            gt_dir = '../PanSharp_dataset/%s/RR/GT.tif'%dataset
            # ref_dir = '../PanSharp_dataset/%s/test/REF_FR/'%dataset
            
            sr_files  = glob.glob(os.path.join(sr_dir, '*.npy'))
            metrics = None
            for idx_img, path_img in enumerate(sr_files):
                sr = np.load(path_img).astype(np.float32)
                name_sr = os.path.basename(path_img).replace('SR', '')
                # gt = np.load(os.path.join(gt_dir, name_sr)).astype(np.float32)
                gt = imageio.imread(gt_dir)
                tmp_metrics = pan_calc_metrics_all({'OUT':sr, 'GT':gt, 'SENSOR':dataset}, 4, img_range, False)
                if metrics is None: metrics = {key: [] for key in tmp_metrics.keys()}
                for key, value in tmp_metrics.items(): metrics[key].append(value)
                # print('%s: %s'%(method, tmp_metrics))
            ave_metrics = {key:sum(value)/len(value) for key, value in metrics.items()}
            print('%s |%s | Average :%s'%(dataset, method, ave_metrics))
            ave_metrics['methods']=method
            ave_metrics['datasets']=dataset
            with open('reduce_results.csv', 'a') as f:
                w = csv.DictWriter(f, ave_metrics.keys())
                w.writeheader()
                w.writerow(ave_metrics)
    # print(ave_metrics)
        