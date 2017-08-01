import torch.utils.data
import torch
import numpy as np 
from PIL import Image
from libs.utils import utils
import pandas as pd
import os

def load_label_map(file):
    with open(file, 'r') as f:
        label = f.readlines()
        label = [i[:-2].split(',') for i in label]

    label_map = {int(i[0]):i[1] for i in label}
    inv_label_map = {i[1]:int(i[0]) for i in label}

    return label_map, inv_label_map

def make_dataset(data_dir, data_csv, split_name, num_classes, inv_label_map):
    if split_name not in ['test', 'eval', 'train']:
        raise ValueError('split name must be test train or eval')

    imgs = []
    labels = None
    df_labels = pd.read_csv(data_csv)
    img_names = df_labels['image_name'].values
    num_imgs = len(img_names)

    if split_name in ['train', 'eval']:
        img_labels = df_labels['tags'].values

        if split_name == 'train':
            # img_labels = img_labels[num_imgs/5:]
            # img_names = img_names[num_imgs/5:]
            img_labels = np.concatenate([img_labels[:num_imgs/5],img_labels[num_imgs/5*2:]], axis=0)
            img_names = np.concatenate([img_names[:num_imgs/5],img_names[num_imgs/5*2:]], axis=0)
		    # img_labels = img_labels[:-num_imgs/5]
		    # img_names = img_names[:-num_imgs/5]
        else:
            # img_labels = img_labels[:num_imgs/5]
            # img_names = img_names[:num_imgs/5]
            img_labels = img_labels[num_imgs/5:num_imgs/5*2]
            img_names = img_names[num_imgs/5:num_imgs/5*2]
		    # img_labels = img_labels[-num_imgs/5:]
		    # img_names = img_names[-num_imgs/5:]

        num_imgs = len(img_names)
        labels = np.zeros((num_imgs, num_classes))

        for i in range(num_imgs):
            targets = np.zeros(num_classes)
            for j in img_labels[i].split(' '):
                targets[inv_label_map[j]] = 1
            labels[i] = targets

    for i in range(num_imgs):
        imgs.append(os.path.join(data_dir, img_names[i]+'.jpg'))

    return imgs, labels
class kgforest(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_csv, split_name, \
            transform=None, target_transform=None, label_map_txt=None):
        
        self.data_dir = data_dir
        self.data_csv = data_csv 
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.label_map_txt = label_map_txt

        _, self.inv_label_map = utils.load_label_map(label_map_txt)
        self.num_classes = len(self.inv_label_map)
        self.imgs, self.labels = \
            make_dataset(self.data_dir, self.data_csv, \
                self.split_name, self.num_classes, self.inv_label_map)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            target = self.labels[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, torch.from_numpy(target).float()

        return img

    def __len__(self):
        return len(self.imgs)

def make_dataset_single(data_dir, data_csv, split_name, num_classes, label_map):
    if split_name not in ['test', 'eval', 'train']:
        raise ValueError('split name must be test train or eval')

    imgs = []
    labels = None
    df_labels = pd.read_csv(data_csv)
    img_names = df_labels['image_name'].values
    num_imgs = len(img_names)

    if split_name in ['train', 'eval']:
        img_labels = df_labels['tags'].values

        if split_name == 'train':
            img_labels = img_labels[num_imgs/5:]
            img_names = img_names[num_imgs/5:]
        else:
            img_labels = img_labels[:num_imgs/5]
            img_names = img_names[:num_imgs/5]

        num_imgs = len(img_names)
        labels = np.zeros(num_imgs).astype(np.float32)

        for i in range(num_imgs):
            if label_map[6] in img_labels[i].split(' '):
                labels[i] = 1

    for i in range(num_imgs):
        imgs.append(os.path.join(data_dir, img_names[i]+'.jpg'))

    return imgs, labels

class kgforest_single(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_csv, split_name, \
            transform=None, target_transform=None, label_map_txt=None):
        
        self.data_dir = data_dir
        self.data_csv = data_csv 
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.label_map_txt = label_map_txt

        self.label_map, _ = utils.load_label_map(label_map_txt)
        self.num_classes = 1
        self.imgs, self.labels = \
            make_dataset_single(self.data_dir, self.data_csv, \
                self.split_name, self.num_classes, self.label_map)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            target = self.labels[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        return img

    def __len__(self):
        return len(self.imgs)
