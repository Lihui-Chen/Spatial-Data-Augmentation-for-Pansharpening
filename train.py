import random
import time
import torch
import networks
import options.options as option
from solvers import SRSolver
from data import create_dataloader
from data import create_dataset, trans_data
import os
import numpy as np
from test import validate
from utils.util import print_to_markdwon_table



def main():
    ################  { parse args and exp setting}  ################
    args = option.add_train_args().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpuid)
    opt = option.parse(args)

    
    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    pytorch_seed(seed)
    
    


    ##################### {dataset} #####################
    train_set_opt = []
    valid_set_opt = []
    for dataname, dataset_opt in sorted(opt['datasets'].items()):
        print('===> Dataset: %s <===' % (dataname))
        if 'train' in dataname:
            train_batch_size = dataset_opt['batch_size']
            train_set_opt.append(dataset_opt)  
        elif 'val' in dataname:
            valid_set_opt.append(dataset_opt)
            valid_batch_size = dataset_opt['batch_size']
        else:
            raise NotImplementedError(
                "[Error] Dataset phase [%s] in *.json is not recognized." % dataname)
    train_set = create_dataset(train_set_opt, len(train_set_opt)>1)
    valid_set = create_dataset(valid_set_opt, False)
    
    train_loaders = create_dataloader(train_set, train_batch_size, collate_type = opt['mask_training'],
                                      num_workers=(1 if valid_batch_size ==1 else 4))
    # if train_loaders
    val_loaders = create_dataloader(valid_set, valid_batch_size, shuffle=False, collate_type = opt['mask_training'],
                                      num_workers=(1 if valid_batch_size==1 else 4))
    

    ################## {initialize network, optimizer, loss} #####################
    solver = SRSolver(opt)
    run_device = next(solver.model.parameters()).device
    # solver.model, solver.optimizer = amp.initialize(solver.model, solver.optimizer, opt_level='O0')

     ################  { make local logger }  ################
    option.creat_logger_dir(opt)
    model_graph, params = networks.get_network_description(solver.model)
    model_graph = str2txt(opt['networks']['net_arch'], model_graph, params,opt['logger']['exp_root'])
                
    
    solver_log = solver.get_current_log()
    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']
    mask=None

    ################  { Training Section }  ################
    # gradientScaler = GradScaler()
    last_loss = None
    last_tot_norm = None
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        ###################  Training   ###################
        trainLoss = 0
        imgCount = 0
        start_time = time.time()
        acuSteps = opt['solver']['acuSteps']
        solver.model.train()
        torch.cuda.empty_cache()
        idx_batch = 0
        for idx_loader, train_loader in enumerate(train_loaders):
            print('\n===> [%d/%d] [Netx%d: %s] [Dataset: %s] [lr: %.6f] <===' % (epoch, NUM_EPOCH,
                opt['scale'], opt['networks']['net_arch'], train_loader.dataname, solver.get_current_learning_rate()))
            for batch, mask in train_loader:
                batchdata = trans_data.data2device(batch, run_device)
                if opt['networks']['net_arch'] == 'cu_nets':
                    out, trainLoss = solver.model.optimize_joint_parameters(
                        batchdata)
                    trainLoss += trainLoss.item()*batch['LR'].shape[0]
                    imgCount += batch['LR'].shape[0]
                else: 
                    # out  = solver.model(batchdata, mask, is_degrade=False)
                    out  = solver.model(batchdata, mask)
                    loss = solver.loss(out, batchdata)
                    
                    if isinstance(loss, (list, tuple)): loss, _, _ = loss
                    loss /= acuSteps
                    
                    # gradientScaler.scale(loss).backward()
                    loss.backward()
                    
                    # if last_loss is None: last_loss = loss.item()
                    # if epoch > 50 and loss.item() > 3*last_loss: 
                    #     solver.optimizer.zero_grad()
                    #     print('Skip this iteration')
                    #     continue
                    # else:
                    #     last_loss = loss.item()
                        
                    if last_tot_norm is None:
                        last_tot_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                                for p in solver.model.parameters() if p.grad is not None]))
                    # if epoch>50:
                    torch.nn.utils.clip_grad_norm_(solver.model.parameters(), max_norm=last_tot_norm*3, norm_type=2)
                    last_tot_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                                for p in solver.model.parameters() if p.grad is not None]))
                    if (idx_batch+1) % acuSteps == 0:
                        solver.optimizer.step()
                        solver.optimizer.zero_grad()
                        
                    trainLoss += loss.item()*batch['REF'].shape[0]
                    imgCount  += batch['REF'].shape[0]
                    
                if (idx_batch+1) == 100*acuSteps: break #*len(val_loaders):
                idx_batch += 1
            
        end_time = time.time()
        # del loss, out, batchdata
        trainLoss /= imgCount
        print('[Train]: No.Iter: %d | Avg Train Loss: %.6f | lossType: %s | Time: %.1f'
              % ((idx_batch+1) // acuSteps, trainLoss, opt['solver']['loss_name'], end_time-start_time))
        
       
        ################ {validation} ################
        torch.cuda.empty_cache()
        mean_metrics = None
        num_tot_img = sum([len(tmp) for tmp in val_loaders])
        for idx_loader, val_loader in enumerate(val_loaders):
            s = time.time()
            testMetrics, SRout = validate(solver.model, solver.loss, val_loader, opt['datasets'][val_loader.dataname],
                                        is_visualize_out=(epoch%10==0 or 'debug' in opt['logger']['tag']), is_visGT=(epoch == 1))
            e = time.time()
            if mean_metrics is None: 
                mean_metrics = {key: value*len(val_loader)/num_tot_img for key, value in testMetrics.items()}
            else:
                mean_metrics = {key:value*len(val_loader)/num_tot_img+mean_metrics[key] for key, value in testMetrics.items()}

            ################  { Print dataset Results }  ################
            columns = ['Networks']
            rows = ['%8s'%opt['networks']['net_arch']]
            for key, value in testMetrics.items():
                columns +=  [key]
                rows += ['%.5f'%value]
            columns +=  ['Time']
            rows += ['%.5f'%(e-s)]
            print('[ valid ] The %d-th val set: [%s]'%(idx_loader+1, val_loader.dataname))
            print_to_markdwon_table(columns, [rows])

        ################  { Print mean Results }  ################
        columns = ['Networks']
        rows = ['%8s'%opt['networks']['net_arch']]
        for key, value in mean_metrics.items():
            columns +=  [key]
            rows += ['%.5f'%value]
        columns +=  ['Time']
        rows += ['%.5f'%(e-s)]
        print('[ valid ] mean results of the above [%d] validation sets'%(idx_loader+1))
        print_to_markdwon_table(columns, [rows])

        ################  { local logging }  ################
        testMetrics = mean_metrics
        solver_log['epoch'] = epoch
        epoch_is_best = False
        if solver_log['best_pred'] is None or solver_log['best_pred'] > testMetrics['ERGAS']:
            solver_log['best_pred']  = testMetrics['ERGAS']
            epoch_is_best            = True
            solver_log['best_epoch'] = epoch
        solver_log['records']['train_loss'].append(trainLoss)
        solver_log['records']['lr'].append(solver.get_current_learning_rate())
        testMetrics.pop('time')
        if epoch == 1:
            for tkey, tvalue in testMetrics.items(): solver_log['records'][tkey] = [tvalue] 
        else:
            for tkey, tvalue in testMetrics.items(): solver_log['records'][tkey].append(tvalue)
        print('Best Epoch [%d] [ERGAS: %.4f]' %(solver_log['best_epoch'], solver_log['best_pred']))
        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch_is_best)
        solver.save_current_records()

        ####### { Updating params. rely on epochs }  #######
        solver.update_learning_rate()
        # if (epoch)%math.ceil(opt['networks']['temper_step']*NUM_EPOCH)==0 and 'dynamic' in opt['networks']['int_type'].lower():
        #     solver.model.update_temperature()
        if opt['networks']['net_arch'] in ('arbrpn_dy_norm', 'arbrpn_mtf'):
            if (epoch)%50==0 and 'dynamic' in opt['networks']['int_type'].lower():
                solver.model.update_temperature()
                

    best_records = {key: value[solver_log['best_epoch']-1] for key, value
                    in solver_log['records'].items()}

    print('===> Finished !')
    ################  {end main}  ################
    
def str2txt(name, net_arch_str, n, exp_root):
    net_lines = []
    line = net_arch_str + '\n'
    net_lines.append(line)
    line = '====>Network structure: [{}], with parameters: [{:,d}]<===='.format(name, n)
    net_lines.append(line)
    net_lines = ''.join(net_lines)
    if exp_root is not None and os.path.isdir(exp_root):
        with open(os.path.join(exp_root, 'network_summary.txt'), 'w') as f:
            f.writelines(net_lines)
    return net_lines
    

def pytorch_seed(seed=0):
    print("===> Random Seed: [%d]" % seed)
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()
