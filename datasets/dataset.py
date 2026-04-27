from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import json
import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import copy
import nrrd

class HeNe_datasets(Dataset):
    def __init__(self, path_Data, config, data="train", filter_no_object=True):
        super(HeNe_datasets, self).__init__()
        self.patch_size = [config.input_size_h, config.input_size_w]
        self.filter_no_object = filter_no_object
        self.mode=data
        self.hu_range=config.hu_range_w1
        if data == "train":
            image_folder = "train/images/"
            mask_folder = "train/masks/"
            self.transformer = config.train_transformer
        elif data == "val":
            image_folder = "val/images/"
            mask_folder = "val/masks/"
            self.transformer = config.val_transformer if hasattr(config, 'val_transformer') else config.test_transformer
        elif data == "test":
            image_folder = "test/images/"
            mask_folder = "test/masks/"
            self.transformer = config.test_transformer
        else:
            raise ValueError("Data argument is invalid, must be one of these values: train, val, test!")

        if not os.path.exists(os.path.join(path_Data, image_folder)):
            if data == "val":
                print(f"Warning: Validation folder not found at {os.path.join(path_Data, image_folder)}. Creating empty validation set.")
                self.data = []
                return
            else:
                raise FileNotFoundError(f"Folder {os.path.join(path_Data, image_folder)} not found!")

        images_list = sorted(os.listdir(os.path.join(path_Data, image_folder)))
        
        self.data = []
        self.slice_name = []
        print(f"Preparing datase, get all slice from all nrrd cases for {data}")
        for img_name in images_list:
            if not img_name.endswith(".nrrd"):
                continue
            img_path = os.path.join(path_Data, image_folder, img_name)
            mask_name = img_name.replace(".nrrd", ".seg.nrrd")
            mask_path = os.path.join(path_Data, mask_folder, mask_name)

            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                #Read slice data and mask from nrrd cases.
                print(f"Read all slice data and mask from {img_name} and {mask_name}") 
                # Load NRRD files
                img_data, _ = nrrd.read(img_path)  # (H, W, D)              #min:-1024 max:3071
                msk_data, _ = nrrd.read(mask_path)  # (H, W, D)
                # Process all slices in the volume     

                for d in range(img_data.shape[-1]):
                    img_slice = img_data[:, :, d]  # (H, W)
                    msk_slice = msk_data[:, :, d]  # (H, W)

                    #Filtering slice with no object by filter_no_object flag
                    if self.filter_no_object is True and np.sum(msk_slice > 0.5) == 0:
                        continue
                    self.data.append([img_slice, msk_slice])
                    self.slice_name.append(img_name.replace(".nrrd", f"_{d}.png"))
            else:
                print(f"Warning: cannot find image or mask for sample {img_name}.")
    
    def __getitem__(self, idx):
        data=copy.deepcopy(self.data[idx])
        img_slice = data[0]
        msk_slice = data[1]
        if self.mode == "test":
            #For visualize
            img_slice_copy=np.clip(copy.deepcopy(img_slice), self.hu_range[0], self.hu_range[1])
            img_slice_copy = np.clip((img_slice_copy - self.hu_range[0]) / (self.hu_range[1]-self.hu_range[0]) * 255, 0, 255).astype(np.uint8)


        # Add channel dimension
        img_slice = np.expand_dims(img_slice, axis=0)  # (1, H, W)
        msk_slice = np.expand_dims(msk_slice, axis=0)  # (1, H, W)                  #unique: 0: background, 1: tumor, 2: cyst
        
        # Apply transformations if any
        if self.transformer is not None:
            img_slice, msk_slice = self.transformer((img_slice, msk_slice))

        if self.mode == "test":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            return img_slice, msk_slice, self.slice_name[idx],img_slice_copy
        else:
            return img_slice, msk_slice
        
    def __len__(self):
        return len(self.data)
    

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
