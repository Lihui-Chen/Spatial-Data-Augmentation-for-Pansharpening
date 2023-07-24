# Spatial-Data-Augmentation-4-Pansharpening
The official codes of "Spatial Data Augmentation: Improving the Generalization of Neural Networks for Pansharpening"

The data augmentations pipeline is in the function [self.\_\_getitem\_\_()](data/dataset_ram_ori.py), Class LRHRDataset, dataset_ram_ori.py.

One can use args (e.g., is_degrade, scale_change) in the command line or in train.yml to control if adopt the random anisotropic MTF degradation and random GSD rescaling in the training data or not.

A demo of the comand line for training.
```
train.py -net_arch arbrpn_plus -opt options/train/demo.yml -log_dir vscode_debug/ -setmode LRHR -gpuid 0, -get_MTF true -batch_size 8 -repeat 3 -patch_size 16 -scale_delta 0.2 -low_thre 0.5 -high_thre 1 -norm_type global_mean_std -scope img -is_srf false -is_degrade true -scale_change true -num_res 9 -num_cycle 5 -temper_step 0.1 -loss_name selfloss -lr_scheme warm_up -mask_training mask_collate_fn_arbrpn -tag debug -acuSteps 1
```


For more descriptions of the args please see the demo.yml under options/train/demo.yml

The folder tree for datasets is suggested as follows
```
dataset
    --dataset1
        --train
            --LR
            --REF
            --GT
            --REF_FR
        --valid
            --LR
            --REF
            --GT
            --REF_FR
        --test
            --LR
            --REF
            --GT
            --REF_FR
    ...
    --datasetN
        --train
            --LR
            --REF
            --GT
            --REF_FR
        --valid
            --LR
            --REF
            --GT
            --REF_FR
        --test
            --LR
            --REF
            --GT
            --REF_FR
```

If you find the codes helpful in your research, please kindly cite the following paper,
```latex
@ARTICLE{10082993,
  author={Chen, Lihui and Vivone, Gemine and Nie, Zihao and Chanussot, Jocelyn and Yang, Xiaomin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Spatial Data Augmentation: Improving the Generalization of Neural Networks for Pansharpening}, 
  year={2023},
  volume={61},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2023.3262262}}
```