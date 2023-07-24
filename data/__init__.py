import torch.utils.data as data
import torch.utils.data._utils.collate as collate
import data.my_collate_fn as my_collate_fn
# from data.dataset_ram import LRHRDataset
from data.dataset_ram_ori import LRHRDataset
# from data.dataset_ram_FR import LRHRDataset as FRdataset
# from data.dataset_ram_offline import LRHRDataset as offDataset
from data.trans_data import multidata
import torch


class data_prefetcher():
    def __init__(self, loader, img_range=255.0):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.img_range = torch.cuda.FloatTensor([img_range/255.0])
        # self.scale = scale
        self.preload()

    def preload(self):
        try:
            self.nextbatch = next(self.loader)
        except StopIteration:
            self.nextbatch = None
            return
        with torch.cuda.stream(self.stream):
            self.nextbatch[0]['LR'] = self.nextbatch[0]['LR'].cuda(non_blocking=True)#.mul_(self.img_range)
            self.nextbatch[0]['GT'] = self.nextbatch[0]['GT'].cuda(non_blocking=True)#.mul_(self.img_range)
            self.nextbatch[0]['REF'] = self.nextbatch[0]['REF'].cuda(non_blocking=True)#.mul_(self.img_range)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.nextbatch
        self.preload()
        return batch


##################### {data loader} #####################
def make_dataset(opt): 
    
    collate_fn = None
    if opt['mask_training'] is not None: collate_fn = opt['mask_training']
    
    train_set_opt = []
    valid_set_opt = []
    
    for dataname, dataset_opt in sorted(opt['datasets'].items()):
        print('===> Dataset: %s <===' % (dataname))
        if 'train' in dataname:
            train_set_opt.append(dataset_opt)  
        elif 'val' in dataname:
            valid_set_opt.append(dataset_opt)
        else:
            raise NotImplementedError(
                "[Error] Dataset phase [%s] in *.json is not recognized." % dataname)
            
    train_set = create_dataset(train_set_opt, True)
    valid_set = create_dataset(valid_set_opt, False)
    
    train_loaders = create_dataloader(train_set, opt['train_batch_size'], num_workers=(1 if opt['valid_batch_size']==1 else 4))
    valid_loaders = create_dataloader(valid_set, opt['valid_batch_size'], num_workers=(1 if opt['valid_batch_size']==1 else 4))
    return train_loaders, valid_loaders


##################### {dataset} #####################
def create_dataset(dataset_opt:list, is_Catset=False): 
    dataset = []
    for tmpopt in dataset_opt:
        if tmpopt['settype']=='FR':
            dataset.append(FRdataset(tmpopt))
        elif 'offline' == tmpopt['settype']:
            dataset.append(offDataset(tmpopt))
        elif 'pair' == tmpopt['settype']:
            from data.dataset_pairs import LRHRDataset as pairdataset
            dataset.append(pairdataset(tmpopt))
        else:
            dataset.append(LRHRDataset(tmpopt))
    assert len(dataset)>0, 'The dataset_opt is null to creat dataset.'
    if is_Catset:
        dataset = data.ConcatDataset(dataset)
        dataset.name="CatSet"
        return [dataset]
    else:
        return dataset
    
##################### {data loader} #####################
@multidata
def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, collate_type=None):
    
    if collate_type == 'mask_collate_fn_arbrpn':
        collate_fn = my_collate_fn.mask_collate_fn_arbrpn
    elif collate_type == 'mask_collate_fn_trans':
        collate_fn = my_collate_fn.mask_collate_fn_trans
    elif collate_type == 'rand_band_collate_fn':
        collate_fn = my_collate_fn.rand_band_collate_fn
    else: 
        # if collate_type is None:
        collate_fn = collate.default_collate
        # raise('The type of collate_fn with value of [%s] is not recognized'%collate_type)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    loader.dataname = dataset.name
    return loader