import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from .utils import recursive_glob
from augmentations import *
from data.base_dataset import BaseDataset
import PIL

PARAMETER_MAX = 10

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img), None


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v), v


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v), v


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v), v


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img, xy


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img), None


def Identity(img, **kwarg):
    return img, None


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img), None


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v), v


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v, resample=Image.BILINEAR, fillcolor=(127,127,127)), v

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v), v


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v), 256 - v


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), threshold


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Sharpness, 0.9, 0.05),
            (Solarize, 256, 0)]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img, type='crc'):
        aug_type = {'Hflip':False, 'ShearX':1e4, 'ShearY':1e4, 'TranslateX':1e4, 'TranslateY':1e4, 'Rotate':1e4, 'CutoutAbs':1e4}
        if type == 'cr' or type == 'crc':
            ops = random.choices(self.augment_pool, k=self.n)
            for op, max_v, bias in ops:
                v = np.random.randint(1, self.m)
                if random.random() < 0.5:
                    img, params = op(img, v=v, max_v=max_v, bias=bias)
                    if op.__name__ in ['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']:
                        aug_type[op.__name__] = params
        return img, aug_type


def apply_aug(X, aug_param, mode="nearest"):
    if type(X) != torch.Tensor:
        X = torch.Tensor(X)
    dtype = X.dtype
    if 'RandomSized' in aug_param.keys():
        w, h = aug_param['RandomSized']['w'], aug_param['RandomSized']['h']
        if len(X.shape) == 2:
            if mode == 'bilinear':
                X = torch.nn.functional.interpolate(X[None,None].float(), (h,w), mode=mode, align_corners=True).squeeze().type(dtype)
            else:
                X = torch.nn.functional.interpolate(X[None,None].float(), (h,w), mode=mode).squeeze().type(dtype)
        elif len(X.shape) == 3:
            if mode == 'bilinear':
                X = torch.nn.functional.interpolate(X[None].float(), (h,w), mode=mode, align_corners=True).squeeze().type(dtype)
            else:
                X = torch.nn.functional.interpolate(X[None].float(), (h,w), mode=mode).squeeze().type(dtype)
    if 'RandomCrop' in aug_param.keys():
        x1, y1, x2, y2 = aug_param['RandomCrop']['x1'], aug_param['RandomCrop']['y1'], aug_param['RandomCrop']['x2'], aug_param['RandomCrop']['y2']
        X = X[...,y1:y2, x1:x2]
    if 'RandomHorizontallyFlip' in aug_param.keys() and aug_param['RandomHorizontallyFlip']:
        X = torch.Tensor(np.array(X)[...,::-1].copy()).type(dtype)
    assert X.dtype == dtype
    return X

class Cityscapes_loader(BaseDataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [
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
        augmentations = None,
    ):
        """__init__

        :param cfg: parameters of dataset
        :param writer: save the result of experiment
        :param logger: logging file
        :param augmentations: 
        """
        
        self.cfg = cfg
        self.root = cfg.rootpath
        self.labelpath = cfg.labelpath
        self.split = cfg.split
        self.is_transform = cfg.is_transform
        self.augmentations = augmentations
        self.img_norm = cfg.img_norm
        self.use_bgr = cfg.use_bgr
        self.n_classes = 19
        self.img_size = (cfg.img_cols, cfg.img_rows)
        self.mean = np.array(cfg.mean)
        self.std = np.array(cfg.std)
        self.files = {}
        self.paired_files = {}
        self.randaug = RandAugmentMC(2, 10) 

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split) 

        self.files = recursive_glob(rootdir=self.images_base, suffix=".png")# [:20] # find all files from rootdir and subfolders with suffix = ".png"

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [ 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))  

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
        img_path = self.files[index].rstrip()
        if self.labelpath:
            lbl_path = os.path.join(
                self.labelpath,
                os.path.basename(img_path)[:-15] + "leftImg8bit.png",
            )
        else:
            lbl_path = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
                )

        imgpil = img = Image.open(img_path)
        lblpil = lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        if not self.labelpath:
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8)) # 伪标签不需要再处理
        

        aug_params = None
        if self.augmentations:
            img, lbl, aug_params = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        imgpil, params = self.randaug(imgpil)
        mask = torch.zeros(lbl.shape).bool()

        img_aug2 = np.array(imgpil.resize(self.img_size, Image.BILINEAR), dtype=np.uint8)
        h, w = lbl.shape
        v = int(0.2*w)
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))
        
        if aug_params:
            mask = apply_aug(mask, aug_params).bool()
        mask[y0:y1,x0:x1] = True

        img_aug2 = torch.Tensor(img_aug2).permute((2,0,1))
        if aug_params:
            img_aug2 = apply_aug(img_aug2, aug_params, mode="bilinear")
        masked_img = img_aug2 * (~mask[None,:,:])
        if self.img_norm:
            masked_img = masked_img / 255.0
            masked_img -= torch.Tensor(self.mean)[:,None,None]
            masked_img /= torch.Tensor(self.std)[:,None,None]


        batch_dict = {
            'image': img,
            'label': lbl,
            'file_name': self.files[index],
            #'aug_params': aug_params,
            'mask': mask,
            'masked_img':masked_img
        }

        return batch_dict
    

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """

        img = np.array(img)
        if self.use_bgr:
            img = img[:, :, ::-1]
        img = img.astype(np.float64)
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
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes): 
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
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

