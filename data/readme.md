Pipeline of loading data for dataset. Noteworthy, "my_collate_fn.py" is used for ArbRPN series only.

1. Data loading (fileio.py)
   1. get image path, online loading
   2. get path and reading, preload
   
2. Data degradation (augmentation/blur, noise, spe_degradation, downsampling)
   1. online degradation
   2. offline degradation
   if data already exist: load data; otherwise: online augmentation + save augmentated data
3. Data Augmentation Pipeline
   1. online augmentation
   2. offline augmentation: 
   if data already exist: load data; otherwise: online augmentation + save augmentated data
4. Data package
   1. type for dict, such as, dict, list, tuple
6. datatype transfer, device transfer
   1. np2tensor, tensor2np, cpu2gpu, gpu2cpu, todevice
7. dataset:
   1. FR dataset, {LR, REF} maybe with mtf and srf
   2. RR dataset, {LR, REF, GT} maybe with mtf and srf
   


