from typing import *
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score,accuracy_score
import time
from scipy.spatial.distance import cdist
from medpy import metric
#import miseval
from PIL import Image
from utils import save_imgs
from pdb import set_trace as st
import torch.nn.functional as F


def calculate_metric_percase(pred, gt):

    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    specificity=metric.binary.specificity(pred, gt)
    accuracy = accuracy_score(gt.flatten(), pred.flatten())
    if gt.sum() == 0 and pred.sum() == 0:
        f2 = 1.0  
        recall = 1.0
        precision= 1.0
    else:
        recall = metric.binary.recall(pred, gt)
        f2 = fbeta_score(gt.flatten(), pred.flatten(), beta=2, zero_division=0)
        precision=metric.binary.precision(pred, gt)
    
    
    tolerance=1.0
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        iou = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        pred_points = np.argwhere(pred)
        gt_points = np.argwhere(gt)
        if len(pred_points) > 0 and len(gt_points) > 0:
            distances = cdist(pred_points, gt_points)
            min_distances = np.min(distances, axis=1)
            nsd = np.sum(min_distances < tolerance) / len(min_distances)
        else:
            nsd = 0  # Avoid division by zero
    
    elif pred.sum() > 0 and gt.sum() == 0:
        hd95 = np.nan  # No ground truth to compare to
        nsd = 0  # No meaningful NSD calculation
        dice = 0
        iou = 0
    
    elif pred.sum() == 0 and gt.sum() > 0:
        hd95 = np.nan  # No prediction to compare to
        nsd = 0  # No meaningful NSD calculation
        dice = 0
        iou = 0
    
    else:
        hd95 = 0  # No distance since both are empty
        nsd = 1  # Perfect match when both are empty
        dice = 1
        iou = 1


    return dice, iou, hd95, nsd, recall, specificity, precision, f2, accuracy


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    elapsed_time,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 

    loss_list = []

    for iter, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        step += iter
        model.zero_grad()
        images, targets = data
        
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        
        start_time = time.time()                                                        #Train pure time
        out = model(images) # Shape: bs x num_classes x H x W
        end_time = time.time()                                                          #Train pure time
        elapsed_time['train pure time']=elapsed_time['train pure time']+(end_time-start_time)     #Train time

        # Thay đổi kích thước của targets
        out = out.squeeze(1)
        targets = targets.squeeze(1)  # Loại bỏ chiều có kích thước 1
        # Shape: bs x H x W, classes: [0., 1., 2., 3., 4., 5., 6.]]

        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'Train - epoch {epoch}, iter:{iter}, loss:{np.mean(loss_list):.5f}, lr:{now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step

def val_one_epoch(val_loader, model, criterion, epoch, logger, config,elapsed_time, visualize:bool=False) -> Union[float, Tuple[float, List[str], List[np.ndarray], List[np.ndarray]]]:
    model.eval()
    loss = torch.zeros(1, device="cuda")
    metric_list_per_class = {}
    num_classes = config.num_classes

    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data
            img, msk = (
                img.cuda(non_blocking=True).float(),
                msk.cuda(non_blocking=True).float(),
            )

            start_time = time.time()                                                                    #Val pure time
            out = model(img)
            end_time = time.time()                                                                      #Val pure time
            elapsed_time['val pure time']=elapsed_time['val pure time']+(end_time-start_time)           #Val pure time

            # Resize targets to remove channel dimension
            out = out.squeeze(1)  # Shape: [B, H, W]
            msk = msk.squeeze(1)  # Shape: [B, H, W]

            loss += criterion(out, msk)

            # Store predictions and ground truths
            pred = torch.argmax(out, dim=1).cpu().detach().numpy()  # Shape: [BS,H, W]
            msk = msk.cpu().detach().numpy()  # Shape: [BS,H, W]

            for i in range(1, num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice, iou, hd95, nsd, recall, specificity, precision, f2, accuracy = calculate_metric_percase(pred_class, gt_class)
                if i not in metric_list_per_class:
                    metric_list_per_class[i] = []
                metric_list_per_class[i].append([dice, iou, hd95, nsd, recall, specificity, precision, f2, accuracy])

            
    metric_list = []
    for i in range(1, num_classes):
            metric_list_per_class[i] = np.array(metric_list_per_class[i])     
            metric_list_per_class[i] = np.nanmean(metric_list_per_class[i], axis=0)
            metric_list.append([metric_list_per_class[i][0], 
                                metric_list_per_class[i][1],
                                metric_list_per_class[i][2],
                                metric_list_per_class[i][3],
                                metric_list_per_class[i][4],
                                metric_list_per_class[i][5],
                                metric_list_per_class[i][6],
                                metric_list_per_class[i][7],
                                metric_list_per_class[i][8]])

    mean_dice = [metric_list[i][0] for i in range(len(metric_list))]  
    mean_iou = [metric_list[i][1] for i in range(len(metric_list))]  
    mean_hd95 = [metric_list[i][2] for i in range(len(metric_list))]
    mean_nsd = [metric_list[i][3] for i in range(len(metric_list))]
    mean_recall = [metric_list[i][4] for i in range(len(metric_list))]
    mean_specificity = [metric_list[i][5] for i in range(len(metric_list))]
    mean_precision = [metric_list[i][6] for i in range(len(metric_list))]
    mean_f2 = [metric_list[i][7] for i in range(len(metric_list))]
    mean_accuracy = [metric_list[i][8] for i in range(len(metric_list))]
    

    loss /= len(val_loader)

    for i in range(1, num_classes): 
        msg=f"Val - epoch {epoch}, Class {config.label_name[i]}, mean_dice: {mean_dice[i-1]:.5f}, mean_iou: {mean_iou[i-1]:.5f}, mean_hd95: {mean_hd95[i-1]:.5f}, mean_nsd: {mean_nsd[i-1]:.5f}, mean_recall: {mean_recall[i-1]:.5f}, mean_specificity: {mean_specificity[i-1]:.5f}, mean_precision: {mean_precision[i-1]:.5f}, mean_f2: {mean_f2[i-1]:.5f}, mean_accuracy: {mean_accuracy[i-1]:.5f}"
        print(msg)
        logger.info(msg)

    all_class_mean_dice=np.mean(mean_dice)
    all_class_mean_iou=np.mean(mean_iou)
    all_class_mean_hd95=np.mean(mean_hd95)
    all_class_mean_nsd=np.mean(mean_nsd)
    all_class_mean_recall=np.mean(mean_recall)
    all_class_mean_specificity=np.mean(mean_specificity)
    all_class_mean_precision=np.mean(mean_precision)
    all_class_mean_f2=np.mean(mean_f2)
    all_class_mean_accuracy=np.mean(mean_accuracy)

    msg="ALL class: mean_dice: %.5f, mean_iou: %.5f, mean_hd95: %.5f, mean_nsd: %.5f, mean_recall: %.5f, mean_specificity: %.5f, mean_precision: %.5f, mean_f2: %.5f, mean_accuracy: %.5f"% (all_class_mean_dice,all_class_mean_iou,all_class_mean_hd95,all_class_mean_nsd,all_class_mean_recall,all_class_mean_specificity,all_class_mean_precision,all_class_mean_f2,all_class_mean_accuracy)
    print(msg)
    logger.info(msg)

    return loss.item(), all_class_mean_dice, all_class_mean_iou, all_class_mean_hd95, all_class_mean_nsd, all_class_mean_recall, all_class_mean_specificity, all_class_mean_precision, all_class_mean_f2, all_class_mean_accuracy


def test_one_epoch(test_loader, model, criterion, logger, config,elapsed_time,visualize=0):
    """
    Testing for one epoch.
    """
    visualize_name=[]
    visualize_ouput=[]
    visualize_gt=[]
    visualize_slice=[]
    metric_list_per_class = {}
    num_classes = config.num_classes
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk, slice_name, img_slice_vis = data
            img, msk = (
                img.cuda(non_blocking=True).float(),
                msk.cuda(non_blocking=True).long(),
            )

            start_time = time.time()                                                                        #Test pure time
            out = model(img)
            end_time = time.time()                                                                          #Test pure time
            elapsed_time['test pure time']=elapsed_time['test pure time']+(end_time-start_time)             #Test pure time

            # Resize targets to remove channel dimension
            msk = msk.squeeze(1)  # Shape: [B, H, W]
            out = out.squeeze(1)  # Shape: [B, H, W]
            
            # Store predictions and ground truths
            msk = msk.cpu().detach().numpy()  # Shape: [BS,H, W]
            pred = torch.argmax(out, dim=1).cpu().detach().numpy()  # Shape: [BS,H,W]    
            
            assert pred.shape[0]==1, "Test just run on batch-size of 1"
            #Store for visualize
            if visualize>0:
                vis=[pred[i] for i in np.arange(pred.shape[0])]
                gt=[msk[i] for i in np.arange(msk.shape[0])]
                visualize_ouput.extend(vis)
                visualize_gt.extend(gt)
                visualize_name.extend(slice_name)
                visualize_slice.extend(img_slice_vis)

            for i in range(1, num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice, iou, hd95, nsd, recall, specificity, precision, f2, accuracy = calculate_metric_percase(pred_class, gt_class)
                if i not in metric_list_per_class:
                    metric_list_per_class[i] = []
                metric_list_per_class[i].append([dice, iou, hd95, nsd, recall, specificity, precision, f2, accuracy])
                
    metric_list = []
    for i in range(1, num_classes):
            metric_list_per_class[i] = np.array(metric_list_per_class[i])     
            metric_list_per_class[i] = np.nanmean(metric_list_per_class[i], axis=0)
            metric_list.append([metric_list_per_class[i][0], 
                                metric_list_per_class[i][1],
                                metric_list_per_class[i][2],
                                metric_list_per_class[i][3],
                                metric_list_per_class[i][4],
                                metric_list_per_class[i][5],
                                metric_list_per_class[i][6],
                                metric_list_per_class[i][7],
                                metric_list_per_class[i][8]])

    mean_dice = [metric_list[i][0] for i in range(len(metric_list))]  
    mean_iou = [metric_list[i][1] for i in range(len(metric_list))]  
    mean_hd95 = [metric_list[i][2] for i in range(len(metric_list))]
    mean_nsd = [metric_list[i][3] for i in range(len(metric_list))]
    mean_recall = [metric_list[i][4] for i in range(len(metric_list))]
    mean_specificity = [metric_list[i][5] for i in range(len(metric_list))]
    mean_precision = [metric_list[i][6] for i in range(len(metric_list))]
    mean_f2 = [metric_list[i][7] for i in range(len(metric_list))]
    mean_accuracy = [metric_list[i][8] for i in range(len(metric_list))]

    for i in range(1, num_classes): 
        msg=f"Test - Class {config.label_name[i]}, mean_dice: {mean_dice[i-1]:.5f}, mean_iou: {mean_iou[i-1]:.5f}, mean_hd95: {mean_hd95[i-1]:.5f}, mean_nsd: {mean_nsd[i-1]:.5f}, mean_recall: {mean_recall[i-1]:.5f}, mean_specificity: {mean_specificity[i-1]:.5f}, mean_precision: {mean_precision[i-1]:.5f}, mean_f2: {mean_f2[i-1]:.5f}, mean_accuracy: {mean_accuracy[i-1]:.5f}"
        print(msg)
        logger.info(msg)

    all_class_mean_dice=np.mean(mean_dice)
    all_class_mean_iou=np.mean(mean_iou)
    all_class_mean_hd95=np.mean(mean_hd95)
    all_class_mean_nsd=np.mean(mean_nsd)
    all_class_mean_recall=np.mean(mean_recall)
    all_class_mean_specificity=np.mean(mean_specificity)
    all_class_mean_precision=np.mean(mean_precision)
    all_class_mean_f2=np.mean(mean_f2)
    all_class_mean_accuracy=np.mean(mean_accuracy)

    msg="ALL class: mean_dice: %.5f, mean_iou: %.5f, mean_hd95: %.5f, mean_nsd: %.5f, mean_recall: %.5f, mean_specificity: %.5f, mean_precision: %.5f, mean_f2: %.5f, mean_accuracy: %.5f"% (all_class_mean_dice,all_class_mean_iou,all_class_mean_hd95,all_class_mean_nsd,all_class_mean_recall,all_class_mean_specificity,all_class_mean_precision,all_class_mean_f2,all_class_mean_accuracy)
    print(msg)
    logger.info(msg)

    return visualize_name, visualize_ouput,visualize_gt, visualize_slice