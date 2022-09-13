# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
from numpy.lib.arraysetops import isin
import torch
import numpy as np
import torchvision.transforms.functional as tf
import torch.nn.functional as F


from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        if img.size != mask.size:
            print (img.size, mask.size)
        assert img.size == mask.size

        aug_params = {}
        for a in self.augmentations:
            img, mask, p = a(img, mask)
            if p:
                aug_params.update(p)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8) 
        
        return img, mask, aug_params


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask, None


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask,):
        assert img.size == mask.size
        return tf.adjust_saturation(img, 
                                    random.uniform(1 - self.saturation, 
                                                   1 + self.saturation)), mask, None


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, 
                                                  self.hue)), mask, None


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, 
                                    random.uniform(1 - self.bf, 
                                                   1 + self.bf)), mask, None

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, 
                                  random.uniform(1 - self.cf, 
                                                 1 + self.cf)), mask, None


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    {'RandomHorizontallyFlip':True}
                )

        return img, mask, {'RandomHorizontallyFlip':False}

class RandomCrop(object):
    def __init__(self, width, height, max_ratio=0.75):
        self.size = (width, height)
        self.max_ratio = max_ratio

    def __call__(self, img, mask):

        assert img.size == mask.size
        w, h = img.size
        tw, th = self.size

        if w == tw and h == th:
            return img, mask, {'RandomCrop':{'x1':0, 'y1':0, 'x2':0+tw, 'y2':0+th}}
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
                {'RandomCrop':{'x1':0, 'y1':0, 'x2':0+tw, 'y2':0+th}}
            )

        for i in range(20):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            mask_crop = mask.crop((x1, y1, x1 + tw, y1 + th))
            ratio = np.bincount(np.array(mask_crop).flatten()).max() / (tw * th)
            if ratio < self.max_ratio:
                break
        
        img_crop = img.crop((x1, y1, x1 + tw, y1 + th))
        return img_crop, mask_crop, {'RandomCrop':{'x1':x1, 'y1':y1, 'x2':x1+tw, 'y2':y1+th}}

class RandomSized(object):
    def __init__(self, size, min=0.5, max=1.5):
        self.size = size
        self.min = min
        self.max = max

    def __call__(self, img, mask):
        assert img.size == mask.size

        prop = 1.0 * img.size[0] / img.size[1]
        w = int(random.uniform(self.min, self.max) * self.size) 
        h = int(w/prop)

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return img, mask, {'RandomSized':{'w':w, 'h':h}}

