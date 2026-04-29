import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric

import numpy as np
from PIL import Image



def visualize_mask_0(mask_output, save_path,image_slice): 
    # Read original image to get target size
    #img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    img = np.array(Image.fromarray(image_slice.detach().cpu().numpy(), mode="L").convert("RGB"), dtype=np.uint8)
    img_h, img_w = img.shape[:2]

    # Xử lý mask_output:
    # Giả sử mask_output ∈ [0,1], shape = (1, 256, 256) hoặc (1, 1, 256, 256)
    mask_output = np.squeeze(mask_output)  # đảm bảo (H, W)

    # Scale to [0, 255] and convert to PIL image
    mask_resized = Image.fromarray((mask_output * 255).astype(np.uint8)).resize((img_w, img_h), resample=Image.BILINEAR)

    # Convert grayscale mask to RGB by stacking 3 channels
    mask_rgb = np.stack([np.array(mask_resized)] * 3, axis=-1)  # (H, W, 3)

    # Save as RGB image
    mask_image = Image.fromarray(mask_rgb.astype(np.uint8))
    mask_image.save(save_path)

def visualize_mask_1(color_palette,mask_output, save_path, image_slice, opacity = 0.3): 
    # --- Rotate input tensors 90° clockwise ---
    image_slice = torch.rot90(image_slice, k=-1, dims=(0, 1))
    mask_output = np.rot90(mask_output, k=-1)

    image_slice = torch.flip(image_slice, dims=[1])   # flip theo chiều ngang (W axis)
    mask_output = np.fliplr(mask_output)              # numpy flip left-right

    img = np.array(
        Image.fromarray(image_slice.detach().cpu().numpy(), mode="L").convert("RGB"),
        dtype=np.uint8
    )
    img_h, img_w = img.shape[:2]

    # Create a color mask based on the color_palette
    mask_h, mask_w = mask_output.shape
    color_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for class_id, color in color_palette.items():
        if class_id != 0:  # Skip background (class_id 0)
            color_mask[mask_output == class_id] = color  

    # Resize the color mask to match the dimensions of the original image
    color_mask = np.array(Image.fromarray(color_mask).resize((img_w, img_h), Image.NEAREST))

    # Create a binary mask to identify the areas that need overlay
    binary_mask = np.any(color_mask > 0, axis=-1)  # True for pixels that have color

    # Create the overlay image by blending only the masked pixels
    overlay = img.copy()
    overlay[binary_mask] = (opacity * color_mask[binary_mask] + (1 - opacity) * img[binary_mask]).astype(np.uint8)

    # Convert the result to a PIL image and save
    Image.fromarray(overlay).save(save_path)

def visualize_mask_2(color_palette,mask_output,mask_gt, save_path,image_slice,opacity = 0.3): 
    img = np.array(
        Image.fromarray(image_slice.detach().cpu().numpy(), mode="L").convert("RGB"),
        dtype=np.uint8
    )
    img_h, img_w = img.shape[:2]


    # Predict
    mask_h, mask_w = mask_output.shape
    color_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for class_id, color in color_palette.items():
        if class_id != 0:  # Skip background (class_id 0)
            color_mask[mask_output == class_id] = color  
    color_mask = np.array(Image.fromarray(color_mask).resize((img_w, img_h), Image.NEAREST))
    binary_mask = np.any(color_mask > 0, axis=-1)  
    overlay_pred = img.copy()
    overlay_pred[binary_mask] = (opacity * color_mask[binary_mask] + (1 - opacity) * img[binary_mask]).astype(np.uint8)


    # Targer
    mask_h, mask_w = mask_gt.shape
    color_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for class_id, color in color_palette.items():
        if class_id != 0:  # Skip background (class_id 0)
            color_mask[mask_gt == class_id] = color  
    color_mask = np.array(Image.fromarray(color_mask).resize((img_w, img_h), Image.NEAREST))
    binary_mask = np.any(color_mask > 0, axis=-1)  
    overlay_target = img.copy()
    overlay_target[binary_mask] = (opacity * color_mask[binary_mask] + (1 - opacity) * img[binary_mask]).astype(np.uint8)

    spacer = 255 * np.ones((img_h, 5, 3), dtype=np.uint8)  # white (H,5,3)
    combined = np.concatenate([overlay_pred, spacer, overlay_target], axis=1)
    Image.fromarray(combined).save(save_path)

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR', 'LambdaLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'LambdaLR':
        var_nepoch=config.epochs
        var_power=config.power
        lr_func = lambda epoch: 1.0 - pow((epoch / var_nepoch), var_power)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')      
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        #pred_ = pred.view(size, -1) # binary isic
        pred_ = pred.reshape(size, -1) # 6 classes vofo
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        
        smooth = 1
        size = pred.size(0)
        #pred_ = pred.view(size, -1) # binary isic
        pred_ = pred.reshape(size, -1) # 6 classes vofo
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size
        return dice_loss


class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i,...], target[:, i,...])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss



class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss


class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        #return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
        return torch.tensor(image).float(), torch.tensor(mask).float()

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        # Resize image with default interpolation (bilinear)
        resized_image = TF.resize(image, [self.size_h, self.size_w],interpolation=InterpolationMode.BILINEAR)
        # Resize mask with nearest interpolation
        resized_mask = TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST)
        return resized_image, resized_mask

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.degree = degree
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            # Rotate image with default interpolation (bilinear)
            rotated_image = TF.rotate(image, angle,interpolation=InterpolationMode.BILINEAR)
            # Rotate mask with nearest interpolation
            rotated_mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            return rotated_image, rotated_mask
        else:
            return image, mask

class myRandomCrop:
    def __init__(self, size):
        """
        Args:
            size (tuple or int): Kích thước crop (h, w) hoặc số int (h=w)
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        image, mask = data
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.size)
        cropped_image = TF.crop(image, i, j, h, w)
        cropped_mask  = TF.crop(mask,  i, j, h, w)
        return cropped_image, cropped_mask

class myRandomScale:
    def __init__(self, scale_range=(0.5, 2.0), p=0.5):
        """
        Args:
            scale_range (tuple): min_scale, max_scale
            p (float): xác suất áp dụng scale
        """
        self.scale_range = scale_range
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            _, h, w = image.size()
            new_h, new_w = int(h * scale), int(w * scale)

            scaled_image = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
            scaled_mask  = TF.resize(mask,  (new_h, new_w), interpolation=InterpolationMode.NEAREST)
            return scaled_image, scaled_mask
        else:
            return image, mask
        
class myNormalize:
    def __init__(self, data_name, hu_range_w1=(-1000, 1000),hu_range_w2=(-1000, 1000),hu_range_w3=(-1000, 1000)):
        self.data_name=data_name
        if data_name == 'MasHeNe_65':
            self.hu_min_w1, self.hu_max_w1 = hu_range_w1
            self.hu_min_w2, self.hu_max_w2 = hu_range_w2
            self.hu_min_w3, self.hu_max_w3 = hu_range_w3

    def __call__(self, data):
        img, msk = data
        if self.data_name == 'MasHeNe_65':
            #Window 1
            img1 = np.clip(img, self.hu_min_w1, self.hu_max_w1)
            img1_normalized= ((img1 - self.hu_min_w1) / (self.hu_max_w1 - self.hu_min_w1))

            #Window 2
            img2 = np.clip(img, self.hu_min_w2, self.hu_max_w2)
            img2_normalized= ((img2 - self.hu_min_w2) / (self.hu_max_w2 - self.hu_min_w2))

            #Window 3
            img3 = np.clip(img, self.hu_min_w3, self.hu_max_w3)
            img3_normalized= ((img3 - self.hu_min_w3) / (self.hu_max_w3 - self.hu_min_w3))

            img_normalized = np.concatenate([img1_normalized, img2_normalized, img3_normalized], axis=0)

        return img_normalized, msk

    



from thop import profile		 ## 导入thop模块
def cal_params_flops(model, size_h,size_w, logger):
    input = torch.randn(1, 1, size_h,size_w).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
