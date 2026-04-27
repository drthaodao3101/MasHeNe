import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import HeNe_datasets
from tensorboardX import SummaryWriter
from models.wemf.wemf import WEMF
import time
from engine import *
import os
import sys
from pathlib import Path

from utils import *
from configs.config_setting import setting_config
import argparse
import warnings
warnings.filterwarnings("ignore")

from pdb import set_trace as st

def main(config):
    parser = argparse.ArgumentParser()           
    parser.add_argument('--mode',type=str,default="train",help="Run mode, can be one of these values:train,val,test (default: 'train')")
    parser.add_argument('--gpu',type=str,default="0",help="Run on which GPU")
    parser.add_argument('--exp',type=str,default="1",help="Run on which which exp")

    parser.add_argument('--visualize', type=int,default="0", help="Enable visualization (0: No visualize, 1: visualize predict only, 2: visualize predict with ground truth)")
    args = parser.parse_args()
    config.gpu_id=args.gpu
    config.work_dir = config.work_dir + f'/exp{args.exp}/'
    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir)

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    #set_seed(config.seed)
    set_seed(int(args.exp))
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    if args.mode=="train":
        train_dataset = HeNe_datasets(config.data_path, config, data='train',filter_no_object=config.filter_no_object_train)
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=config.num_workers)
        val_dataset = HeNe_datasets(config.data_path, config, data='val', filter_no_object=config.filter_no_object_val)
        val_loader = DataLoader(val_dataset,
                                    batch_size=1,                               #Always set to 1
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    #drop_last=False                
                                    )
        test_loader=None

    elif args.mode=="test":
        train_loader = None 
        val_loader = None
        test_dataset = HeNe_datasets(config.data_path, config, data='test',filter_no_object=config.filter_no_object_test)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,                               #Always set to 1
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    #drop_last=True                 
                                    )
    else:
        raise ValueError("Mode argument is invalid, must one of these values:train,test!")

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    
    if config.network == 'wemf':
        model = WEMF(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
        
    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, config.input_size_h,config.input_size_w, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    best_dice = 0
    best_iou = 999
    start_epoch = 1
    min_epoch = 1

    elapsed_time={
        "train + val time":0.0,                             
        "train + val pure time":0.0,
        "train time":0.0,                             
        "train pure time":0.0,                        
        "val time":0.0,                             
        "val pure time":0.0,                        
        "test time":0.0,
        "test time-fps":0.0,
        "test pure time":0.0,
        "test pure time-fps":0.0,
    }
    
    if args.mode=="train":
        patience=config.patience
        patience_count=0

        if os.path.exists(resume_model):
            print('#----------Resume Model and Other params----------#')
            checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            saved_epoch = checkpoint['epoch']
            start_epoch += saved_epoch
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
            best_dice, best_iou,patience_count =  checkpoint['best_dice'], checkpoint['best_iou'],checkpoint['patience_count']
            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
            logger.info(log_info)

        step = 0
        print('#----------Training----------#')
        
        for epoch in range(start_epoch, config.epochs + 1):
            
            torch.cuda.empty_cache()
            start_time = time.time()                                                        #Train time
            step = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                step,
                logger,
                config,
                elapsed_time,
                writer
            )
            end_time = time.time()                                                          #Train time
            elapsed_time['train time']=elapsed_time['train time']+(end_time-start_time)     #Train time

            start_time = time.time()                                                        #Val time
            loss, mean_dice, mean_iou, _, _, _,_, _, _, _ = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config,
                    elapsed_time
                )
            end_time = time.time()                                                          #Val time
            elapsed_time['val time']=elapsed_time['val time']+(end_time-start_time)         #Val time

            # if isinstance(loss, tuple):
                # loss = max(loss)  # Get largest value from the tuple
            if epoch >= config.start_epoch_best_model_saving and mean_dice >= best_dice:
                log_info=f"Saving model with best loss:{loss} and dice:{mean_dice}"
                print(log_info)
                logger.info(log_info)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch
                best_dice = mean_dice
                best_iou = mean_iou
            else:
                patience_count=patience_count+1
                log_info=f"Current best - Loss:{loss}, dice:{best_dice}, iou:{best_iou}"
                print(log_info)
                logger.info(log_info)
                if patience_count>patience:
                    log_info=f"Train is hard stopped because model is not better over {patience} epoches."
                    print(log_info)
                    logger.info(log_info)
                    break
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'best_dice':best_dice,
                    'best_iou':best_iou,
                    'patience_count' :patience_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')) 

    elif args.mode == "test":
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
            model.load_state_dict(best_weight)
            
            start_time = time.time()                                                                        #Test time
            visualize_name, visualize_ouput,visualize_gt, visualize_slice = test_one_epoch(
                    test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    elapsed_time,
                    args.visualize
                )
            
            if args.visualize > 0:
                #Save the mask output to workspace
                visualize_dir_path=os.path.join(config.work_dir,"visualize")
                print(f"Visualize: saving ouput in {visualize_dir_path}")
                folder_path = Path(visualize_dir_path)
                folder_path.mkdir(parents=True, exist_ok=True)
                #Get name without extension
                visualize_name_file=[ os.path.splitext(os.path.basename(p))[0] for p in visualize_name]
                assert len(visualize_ouput)==len(visualize_name_file)
                color_palette=config.color_palette
                for index in np.arange(len(visualize_ouput)):
                    mask_output=visualize_ouput[index]
                    mask_gt=visualize_gt[index]
                    save_path=f"{folder_path}/{visualize_name_file[index]}.png"
                    image_slice = visualize_slice[index]
                    #print(f"{index}-{save_path}")
                    if args.visualize:                                                                 
                        visualize_mask_1(color_palette,mask_output, save_path,image_slice,config.opacity)
                    elif args.visualize == 2:                                                               # visualize predict only
                        visualize_mask_2(color_palette,mask_output,mask_gt, save_path,image_slice,config.opacity)
                    else:
                        raise Exception('Visualize mode in not right! select one in values: 0,1,2')
                    
                
            end_time = time.time()                                                                          #Test time
            elapsed_time['test time']=elapsed_time['test time']+(end_time-start_time)                       #Test time
            elapsed_time['test time-fps']=test_dataset.__len__()/elapsed_time['test time']                  #Test time fps
            elapsed_time['test pure time-fps']=test_dataset.__len__()/elapsed_time['test pure time']    #Test pure time fps
            
  
    else:
        raise ValueError("The argument 'mode' is invalid!")
    
    log_info='#----------Time Summary----------#'
    print(log_info)
    logger.info(log_info)
    
    log_info="Train time only:{elapsed_time['train time']} seconds."
    print(log_info)
    logger.info(log_info)

    log_info=f"Train pure time only:{elapsed_time['train pure time']} seconds."
    print(log_info)
    logger.info(log_info)

    log_info=f"Val time only:{elapsed_time['val time']} seconds."
    print(log_info)
    logger.info(log_info)

    log_info=f"Val pure time only:{elapsed_time['val pure time']} seconds."
    print(log_info)
    logger.info(log_info)

    elapsed_time['train + val time']=elapsed_time['train time']+elapsed_time['val time']

    log_info=f"Train+Val time:{elapsed_time['train + val time']} seconds."
    print(log_info)
    logger.info(log_info)

    elapsed_time['train + val pure time']=elapsed_time['train pure time']+elapsed_time['val pure time']
    
    log_info=f"Train+Val pure time:{elapsed_time['train + val pure time']} seconds."
    print(log_info)
    logger.info(log_info)

    log_info=f"Test time:{elapsed_time['test time']} seconds ({elapsed_time['test time-fps']} fps)."
    print(log_info)
    logger.info(log_info)

    log_info=f"Test pure time:{elapsed_time['test pure time']} seconds ({elapsed_time['test pure time-fps']} fps)."
    print(log_info)
    logger.info(log_info)

if __name__ == '__main__':
    config = setting_config
    main(config)
