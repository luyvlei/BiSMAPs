import os
import random
import torch
import numpy as np
from PIL import Image

from .utils import recursive_glob
from data import BaseDataset
from default.default import cfg as cfgGLobal

from utils.utils import idx_2_color

def apply_aug(X, aug_param):
    X = torch.tensor(X)
    if 'RandomSized' in aug_param.keys():
        w, h = aug_param['RandomSized']['w'], aug_param['RandomSized']['h']
        X = torch.nn.functional.interpolate(X[None,None], (h,w), mode='nearest').squeeze()
    if 'RandomCrop' in aug_param.keys():
        x1, y1, x2, y2 = aug_param['RandomCrop']['x1'], aug_param['RandomCrop']['y1'], aug_param['RandomCrop']['x2'], aug_param['RandomCrop']['y2']
        X = X[y1:y2, x1:x2]
    if 'RandomHorizontallyFlip' in aug_param.keys() and aug_param['RandomHorizontallyFlip']:
        X = torch.Tensor(np.array(X)[:,::-1].copy())
    return X

class GTA5_loader(BaseDataset):
    """
    GTA5    synthetic dataset
    for domain adaptation to Cityscapes
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

        self.mean = np.array(cfg.mean) 
        self.std = np.array(cfg.std)
        self.image_base_path = cfg.imagepath if cfg.imagepath else os.path.join(self.root, 'images')
        self.label_base_path = os.path.join(self.root, 'labels')
        if self.cfg.use_reweight_map:
            self.reweight_map_path = cfg.reweight_map_path
        self.distribute = np.zeros(self.n_classes, dtype=float)
        self.ids = recursive_glob(rootdir=self.label_base_path, suffix=".png")


        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if len(self.ids) == 0:
            raise Exception("No files for style=[%s] found in %s" % (self.split, self.image_base_path))
        
        print("Found {} {} images".format(len(self.ids), self.split))

    def __len__(self):
        """__len__"""
        return len(self.ids) if self.cfg.len is None else self.cfg.len

    def __getitem__(self, index):
        """__getitem__
        
        param: index
        """
        id = self.ids[index]
        if self.split != 'all' and self.split != 'val':
            filename = '{:05d}.png'.format(id)
            img_path = os.path.join(self.image_base_path, filename)
            lbl_path = os.path.join(self.label_base_path, filename)
        else:
            img_path = os.path.join(self.image_base_path, id.split('/')[-1])
            lbl_path = id
                    
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)
        lbl = np.asarray(lbl, dtype=np.uint8)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
        aug_params = None
        if self.augmentations!=None:
            img, lbl, aug_params = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        if self.cfg.use_reweight_map:
            reweight_map_file = os.path.join(self.reweight_map_path, id.split('/')[-1])
            reweight_map = Image.open(reweight_map_file)
            reweight_map = reweight_map.resize(self.img_size, Image.NEAREST)
            reweight_map = np.asarray(reweight_map, dtype=np.uint8)
            reweight_map = apply_aug(reweight_map, aug_params)

        batch_dict = {
            'image': img,
            'label': lbl,
            'file_name': self.ids[index],
            'aug_params': aug_params,
        }
        if self.cfg.use_reweight_map:
            batch_dict['reweight_map'] = reweight_map/255.
        return batch_dict


    def encode_segmap(self, lbl):
        for _i in self.void_classes:
            lbl[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl[lbl == _i] = self.class_map[_i]
        return lbl

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

    def transform(self, img, lbl):
        """transform

        img, lbl
        """
        img = np.array(img)
        if self.use_bgr:
            img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if self.img_norm:
            img = img.astype(float) / 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)
        
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes): 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

