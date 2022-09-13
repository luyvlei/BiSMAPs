import logging
import functools
import inspect

from .loss import cross_entropy2d
from .loss import sceloss


logger = logging.getLogger('ptsemseg')

key2loss = {'cross_entropy': cross_entropy2d,
            'sceloss': sceloss}

def get_loss_function(cfg):
    if cfg.training.loss is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg.training.loss
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k in inspect.getargspec(key2loss[loss_name]).args}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name, 
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
