import os
import torch
import numpy as np
import scipy.misc as m

from PIL import Image

from .utils import recursive_glob
from augmentations import *
from data.base_dataset import BaseDataset

from default.default import cfg as cfgGLobal

import imageio
import random

def apply_aug(X, aug_param):
    X = torch.Tensor(X)
    if 'RandomSized' in aug_param.keys():
        w, h = aug_param['RandomSized']['w'], aug_param['RandomSized']['h']
        X = torch.nn.functional.interpolate(X[None,None], (h,w), mode='nearest').squeeze()
    if 'RandomCrop' in aug_param.keys():
        x1, y1, x2, y2 = aug_param['RandomCrop']['x1'], aug_param['RandomCrop']['y1'], aug_param['RandomCrop']['x2'], aug_param['RandomCrop']['y2']
        X = X[y1:y2, x1:x2]
    if 'RandomHorizontallyFlip' in aug_param.keys() and aug_param['RandomHorizontallyFlip']:
        X = torch.Tensor(np.array(X)[:,::-1].copy())
    return X

class SYNTHIA_loader(BaseDataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
            self,
            cfg,
            augmentations=None,
    ):
        """__init__

        :param cfg: parameters of dataset
        :param writer: save the result of experiment
        :param logger: logging file
        :param augmentations:
        """

        self.cfg = cfg
        self.root = cfg.rootpath
        self.split = cfg.split
        self.is_transform = cfg.get('is_transform', True)
        self.augmentations = augmentations
        self.img_norm = cfg.get('img_norm', True)
        self.use_bgr = cfg.get('use_bgr', False)
        self.n_classes = 19
        self.img_size = (
            cfg.img_cols, cfg.img_rows
        )
        self.files = {}

        self.mean = np.array(cfg.mean) 
        self.std = np.array(cfg.std)
        self.images_base = cfg.imagepath if cfg.imagepath else os.path.join(self.root, 'RGB')
        self.annotations_base = os.path.join(self.root, 'GT', 'LABELS')
        if self.cfg.use_reweight_map:
            self.reweight_map_path = cfg.reweight_map_path

        self.files = recursive_glob(rootdir=self.images_base,
                                    suffix=".png")  # find all files from rootdir and subfolders with suffix = ".png"
        if cfg.get('shuffle'):
            np.random.shuffle(self.files)

        self.distribute = np.zeros(self.n_classes, dtype=float)
        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250

        if not self.files:
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files), self.split))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        filename = self.files[index].rstrip()
        img_path = os.path.join(self.images_base, filename.split('/')[-1])
        lbl_path = os.path.join(self.annotations_base, filename.split('/')[-1])

        img = Image.open(img_path)
        label = np.asarray(imageio.imread(lbl_path, format='PNG-FI'))[:, :, 0] # uint16
        lbl = Image.fromarray(label)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        aug_params = None
        if self.augmentations != None:
            img, lbl, aug_params = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        if self.cfg.use_reweight_map:
            reweight_map_file = os.path.join(self.reweight_map_path, filename.split('/')[-1])
            reweight_map = Image.open(reweight_map_file)
            reweight_map = reweight_map.resize(self.img_size, Image.NEAREST)
            reweight_map = np.asarray(reweight_map, dtype=np.uint8)
            reweight_map = apply_aug(reweight_map, aug_params)

        batch_dict = {
            'image': img,
            'label': lbl,
            'file_name': filename,
            'aug_params': aug_params,
        }
        if self.cfg.use_reweight_map:
            batch_dict['reweight_map'] = reweight_map/255.

        return batch_dict

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = np.array(img)
        if self.use_bgr:
            img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        img -= self.mean
        img /= self.std
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):  # todo: understanding the meaning
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        label_copy = self.ignore_index * np.ones(mask.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[mask == k] = v
        return label_copy
