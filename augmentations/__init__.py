import logging

from PIL import UnidentifiedImageError
from augmentations.augmentations import *

logger = logging.getLogger('ptsemseg')

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'rcrop': RandomCrop,
           'hflip': RandomHorizontallyFlip,
           'rsize': RandomSized,}

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        if aug_param == None:
            continue
        if isinstance(aug_param, list):
            augmentations.append(key2aug[aug_key](*aug_param))
        elif isinstance(aug_param, numbers.Number):
            augmentations.append(key2aug[aug_key](aug_param))
        else:
            raise ValueError
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)
