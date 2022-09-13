from .schedulers import *
import copy

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'cosine_annealing': CosineAnnealingLR,}

key2param = {'constant_lr': [],
                 'poly_lr': ['T_max', 'gamma', 'eta_min'],
                 'cosine_annealing': ['T_max','eta_min'],}

def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        return ConstantLR(optimizer)
    scheduler_dict_old = copy.deepcopy(scheduler_dict)
    s_type = scheduler_dict['name']
    scheduler_dict_old.pop('name')
    warmup_iters = scheduler_dict['warmup_iters']
    scheduler_dict_old.pop('warmup_iters')
    scheduler_dict_old['T_max'] -= warmup_iters
    if s_type == 'cosine_annealing':
        scheduler_dict_old.pop('gamma')

    base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict_old)
    if warmup_iters > 0:
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_iters, after_scheduler=base_scheduler)

        return warmup_scheduler
    return base_scheduler


