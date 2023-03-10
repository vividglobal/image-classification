import torch 
import yaml 
import pandas as pd
import argparse
from utils.torch_utils import select_device,loadingImageNetWeight
from utils.dataset import LoadImagesAndLabels,preprocess
from utils.general import EarlyStoping,visualize
from utils.callbacks import CallBack
from tqdm import tqdm 
import sklearn.metrics
from models import MobileNetV2,resnet18,resnet34,resnet50,resnet101,resnet152,\
                    resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,\
                    wide_resnet101_2,shufflenet_v2_x0_5,shufflenet_v2_x1_0
from models import model_fn,model_urls
import os
import numpy as np
import logging 
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig()
def train(opt):
    softmax = torch.nn.Softmax(1)
    os.makedirs(opt.save_dir, exist_ok=True)
    # read train_csv
    df_train = []
    if isinstance(opt.train_csv,str):
        opt.train_csv = [opt.train_csv]
    for file in opt.train_csv:
        df_train.append(pd.read_csv(file))
    df_train = pd.concat(df_train,axis=0)


    # read val_csv
    df_val = []
    if isinstance(opt.val_csv,str):
        opt.val_csv = [opt.val_csv]
    for file in opt.val_csv:
        df_val.append(pd.read_csv(file))
    df_val = pd.concat(df_val,axis=0)

    device = select_device(opt.device,model_name=opt.model_name)

    if opt.visualize:
        LOGGER.info('data is being visualized, please wait.......')
        visualize(df_train, classes_copy=opt.classes.copy(), format_index=opt.format_index, save_dir=opt.save_dir, dataset_name='train')
        visualize(df_val, classes_copy=opt.classes.copy(), format_index=opt.format_index, save_dir=opt.save_dir, dataset_name='val')

    cuda = device.type != 'cpu'
    ds_train = LoadImagesAndLabels(df_train,
                                data_folder=opt.DATA_FOLDER,
                                img_size = opt.img_size,
                                padding = opt.padding,
                                classes = opt.classes,
                                format_index = opt.format_index,
                                preprocess=preprocess,
                                augment=True,
                                augment_params=opt.augment_params)

    ds_val = LoadImagesAndLabels(df_val,
                                data_folder=opt.DATA_FOLDER,
                                img_size = opt.img_size,
                                padding= opt.padding,
                                classes = opt.classes,
                                format_index = opt.format_index,
                                preprocess=preprocess,
                                augment=False)
    trainLoader = torch.utils.data.DataLoader(ds_train,
                                             batch_size=opt.batch_size,
                                            shuffle=True,)
                                            # num_workers=8)
    valLoader = torch.utils.data.DataLoader(ds_val,
                                            batch_size = opt.batch_size,
                                            shuffle=False,)
                                            # num_workers=8)

    loader = {'train': trainLoader,
              'val'  : valLoader}
    callback = CallBack(opt.save_dir)

    # init model
    model_config = []
    for k,v in opt.classes.items():
        model_config.append(len(v))

    # init model

    model = model_fn[opt.model_name](model_config=model_config)

    if opt.DEBUG: 
        x = np.random.randint(0,255,(1,3,224,224))
        x = x.astype('float')/255.
        print(model.predict(torch.Tensor(x)))

    loss_train_log = []
    loss_val_log = []
    best_fitness,best_epoch = 0,0
    start_epoch = 0
    if os.path.isfile(opt.weights):                    # load from checkpoint
        LOGGER.info(f' loading pretrain from {opt.weights}')
        ckpt_load = torch.load(opt.weights)
        model.load_state_dict(ckpt_load['state_dict'])
        if opt.continue_training:
            start_epoch = ckpt_load['epoch'] + 1
            best_epoch = ckpt_load['best_epoch']
            loss_train_log = ckpt_load['loss_train_log']
            loss_val_log = ckpt_load['loss_val_log']
            fitness = ckpt_load['fitness']
            best_fitness = ckpt_load['best_fitness']
            LOGGER.info(' resume training from last checkpoint')
    else:                                               #load from ImagesNet weight
        LOGGER.info(f"weight path : {opt.weights} does'nt exist, ImagesnNet weight will be loaded ")
        model = loadingImageNetWeight(model,name=opt.model_name,model_urls=model_urls)
       
    model = model.to(device)

    if opt.DEBUG:
        exit()
        
    # optimier
    g0, g1, g2 = [], [], [] #params group  #g0 - BatchNorm, g1 - weight, g2 - bias
    for module in model.modules():
        if hasattr(module,'bias') and isinstance(module.bias, torch.nn.Parameter):
            g2.append(module.bias)
        if isinstance(module, torch.nn.BatchNorm2d):
            g0.append(module.weight)
        elif hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            g1.append(module.weight)
    if hasattr(opt,'adam'):
        optimizer = torch.optim.Adam(g0, lr=opt.hyp['lr0'], betas=opt.hyp['momentum'])
    else:
        optimizer = torch.optim.SGD(g0, lr=opt.hyp['lr0'], momentum=opt.hyp['momentum'])
    optimizer.add_param_group({'params':g1, 'weight_decay': opt.hyp['weight_decay']})
    optimizer.add_param_group({'params':g2})
    del g0,g1,g2

    criteriors = list()
    for label_name in opt.classes:
        if opt.class_weights.get(label_name):
            class_weights = torch.Tensor(opt.class_weights[label_name])
            class_weights = class_weights/class_weights.sum()
            class_weights = class_weights.to(device)
        else: 
            class_weights = None
        criteriors.append(torch.nn.CrossEntropyLoss(weight=class_weights))

    if not isinstance(opt.task_weights,list):
        task_weights = [opt.task_weights]
    
    # if opt.linear_lr:
    #     lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - opt.hyp['lrf']) + opt.hyp['lrf']  # linear
    # else:
    #     # lf = one_cycle(1, opt.hyp['lrf'], epochs)
    #     pass

    stopper = EarlyStoping(best_fitness=best_fitness, best_epoch=best_epoch, patience=opt.patience,ascending=False)
    
    pbar_epoch = tqdm(range(start_epoch,opt.epochs),total=opt.epochs,initial=start_epoch)
    for epoch in pbar_epoch:
        # training phase.
        if epoch!=0 and hasattr(opt,'sampling_balance_data'):
            loader['train'].dataset.on_epoch_end(n=opt.sampling_balance_data)
        
        model.train()
        pbar = enumerate(loader['train'])
        nb = len(loader['train'])
        epoch_loss = 0.0
        warmup_iteration = max(opt.hyp['warmup_epochs']*nb,1000)
        for i, (imgs, labels, _) in tqdm(pbar,total=nb,desc='training',leave=False):
            ni = i + nb * epoch
            imgs = imgs.to(device)
            labels = [label.to(device) for label in labels]
            # Warm-up training
            if ni <= warmup_iteration:
                xi = [0, warmup_iteration]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    #bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [opt.hyp['warmup_bias_lr'] if j == 2 else 0.0, opt.hyp['lr0']]) #x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [opt.hyp['warmup_momentum'], opt.hyp['momentum']])
            
            
            with torch.set_grad_enabled(cuda):
                preds = model(imgs)
                loss = 0
                for index,criterior in enumerate(criteriors):
                    loss += opt.task_weights[index]*criterior(preds[index],labels[index])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                epoch_loss += loss * imgs.size(0)
        epoch_loss = epoch_loss/len(loader['train'].dataset)
        loss_train_log.append(epoch_loss)

        # evaluate phase.
        model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for i, (imgs, labels, _) in tqdm(enumerate(loader['val']),total=len(loader['val']),desc='evaluating',leave=False):
                imgs = imgs.to(device)
                labels = [label.to(device) for label in labels]
                preds = model(imgs)
                loss = 0
                for index,criterior in enumerate(criteriors):
                    loss += opt.task_weights[index]*criterior(preds[index],labels[index])                      
                epoch_loss += loss * imgs.size(0) # imgs.size = batch_size
        epoch_loss = epoch_loss/len(loader['val'].dataset)
        loss_val_log.append(epoch_loss)
        
        # fi - macro avg accuracy

        # fi = sklearn.metrics.classification_report(y_true,y_pred,digits=4,zero_division=1)
        # fi = fi.split('\n')[-3].split()[-2]
        # fi = float(fi)

        fi = epoch_loss.item()

        if stopper(epoch,fi):   #if EarlyStopping condition meet
            break
        if epoch==stopper.best_epoch:
            ckpt_best = { 
                            'state_dict':model.state_dict(),
                            'best_fitness': stopper.best_fitness,
                            'fitness': fi, 
                            'epoch':epoch,
                            'best_epoch': epoch,
                            'loss_train_log': loss_train_log,                  
                            'loss_val_log': loss_val_log,
                            'meta_data': opt
                        }
            torch.save(ckpt_best,os.path.join(opt.save_dir,'best.pt'))
        ckpt_last = { 
                        'state_dict':model.state_dict(),
                        'best_fitness': stopper.best_fitness,
                        'fitness': fi,  
                        'epoch': epoch,
                        'best_epoch': stopper.best_epoch,
                        'loss_train_log': loss_train_log,                  
                        'loss_val_log': loss_val_log,
                        'meta_data': opt
                    }
        torch.save(ckpt_last,os.path.join(opt.save_dir,'last.pt'))   
        callback(loss_train_log[-1],loss_val_log[-1],epoch)   

def parse_opt(know=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',help = 'weight path')
    parser.add_argument('--cfg',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/human_attribute_2/train_config.yaml')
    parser.add_argument('--data',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/human_attribute_2/data_config.yaml')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30, help='patience epoch for EarlyStopping')
    parser.add_argument('--save_dir', type=str, default='', help='save training result')
    parser.add_argument('--task_weights', type=list, default=1, help='weighted for each task while computing loss')
    opt = parser.parse_known_args()[0] if know else parser.parse_arg()
    return opt 

if __name__ =='__main__': 
    opt = parse_opt(True)
    with open(opt.cfg) as f:
        cfg = yaml.safe_load(f)
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    for k,v in cfg.items():
        setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    assert isinstance(opt.classes,dict), "Invalid format of classes in data_config.yaml"
    assert len(opt.task_weights) == len(opt.classes), "task weight should has the same length with classes"
    train(opt)
