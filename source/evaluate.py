import torch
from models.mobilenetv2 import MobileNetV2
import sklearn.metrics 
from tqdm import tqdm 
import argparse
import logging 
from utils.dataset import LoadImagesAndLabels, preprocess
from utils.torch_utils import select_device
import yaml
import pandas as pd
import os
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig()

def evaluate(opt):
    
    if isinstance(opt.val_csv,str):
        opt.val_csv = [opt.val_csv]
    df_val = []
    for df in opt.val_csv:
        df  = pd.read_csv(df)
        df_val.append(df)
    df_val = pd.concat(df_val, axis=0)
    # df_val = df.sample(frac=1).reset_index(drop_index=True)
    model_config = []
    for k,v in opt.classes .items():
        model_config.append(len(v))
    model = MobileNetV2(model_config)
    
    if not os.path.isfile(opt.weights): 
        LOGGER.info(f"{opt.weights} is not a file")
        exit()
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['state_dict'])
    padding = getattr(checkpoint['meta_data'], 'padding')
    img_size = getattr(checkpoint['meta_data'], 'img_size')
    
    
    ds_val = LoadImagesAndLabels(df_val,
                                data_folder=opt.DATA_FOLDER,
                                img_size = img_size,
                                padding= padding,
                                classes = opt.classes,
                                format_index = opt.format_index,
                                preprocess=preprocess,
                                augment=False)
    loader_val = torch.utils.data.DataLoader(ds_val,
                                            batch_size=opt.batch_size,
                                            shuffle=True
                                            )
    loader = {'val':loader_val}
    device = select_device(opt.device, model_name=getattr(checkpoint['meta_data'],'model_name'))
    model = model.to(device)
    model.eval()

    y_true = [] 
    y_pred = [] 
    for _ in range(len(opt.classes)):
        y_true.append([])
        y_pred.append([])
    with torch.no_grad(): 
        for i,(imgs,labels,path) in tqdm(enumerate(loader['val']),total=len(loader['val'])):
            imgs = imgs.to(device)
            preds = model.predict(imgs)
            labels = [label.to(device).cpu().numpy().ravel() for label in labels]
            # LOGGER.info(f'len_labels: {len(labels[0])}')
            preds = [x.detach().cpu().numpy().argmax(axis=-1).ravel() for x in preds]
            for j in range(len(opt.classes)):
                y_true[j].append(labels[j])
                y_pred[j].append(preds[j])
    y_true = [ np.concatenate(x, axis=0) for x in y_true ]
    y_pred = [ np.concatenate(x, axis=0) for x in y_pred ]
    LOGGER.debug(f"y_true[0]_len = {len(y_true[0])}")
    LOGGER.debug(f"y_true[1]_len = {len(y_true[1])}")
    LOGGER.debug(f"y_pred[0]_len = {len(y_pred[0])}")
    LOGGER.debug(f"y_pred[1]_len = {len(y_true[1])}")
    for i,(k,v) in enumerate(opt.classes.items()):
        fi = sklearn.metrics.classification_report(y_true[i],y_pred[i],digits=4,zero_division=1,target_names=v)
        with open(opt.logfile,'a') as f:
            f.write(f'-------------{k}-----------\n')
            f.write(fi+'\n')
            print(f'-------------{k}-----------\n')
            print(fi+'\n')

def parse_opt(know):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help="checkpoint path")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size in evaluating")
    parser.add_argument('--device', type=str, default='', help="select gpu")
    parser.add_argument("--val_csv",type=str, default='',help='')
    # parser.add_argument('--cfg',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/default/train_config.yaml')
    parser.add_argument('--data',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/default/data_config.yaml')
    parser.add_argument("--logfile", type=str, default="log.evaluate.txt", help="log the evaluating result")
    opt = parser.parse_known_args()[0] if know else parser.parse_arg()
    return opt 

def main():
    opt = parse_opt(True)
    # with open(opt.cfg) as f:
        # cfg = yaml.safe_load(f)
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    # for k,v in cfg.items():
        # setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    assert isinstance(opt.classes,dict), "Invalid format of classes in data_config.yaml"
    # assert len(opt.task_weights) == len(opt.classes), "task weight should has the same length with classes"
    evaluate(opt)

if __name__ =='__main__':
    main()


