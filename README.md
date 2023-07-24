# Spatial-Data-Augmentation-4-Pansharpening
The official codes of "Spatial Data Augmentation: Improving the Generalization of Neural Networks for Pansharpening"

The data augmentataion pipeline is in the function [self.\_\_getitem\_\_()](data/dataset_ram_ori.py), Class LRHRDataset, dataset_ram_ori.py.

One can use args (e.g, is_degrade, scale_change) in command line or in train.yml to control if adopt the random anisotropic MTF degradation and random GSD rescaling in the training data or not.

More descriptions of the args please see a the demo.yml under options/train/demo.yml

The folder tree for dataset is suggested as follows
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
