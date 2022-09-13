import torch.utils.data as data
import random
import numpy as np
import torch 

GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

class DataProvider():
    def __init__(self, dataset, **kw):
        self.args = kw
        self.dataset = dataset
        self.epoch = 0
        self.DataLoader = None
        self.iteration = 0
        self.build()
    
    def build(self):
        if self.args['sampler']:
            self.args['sampler'].set_epoch(self.epoch)
        self.DataLoader = data.DataLoader(self.dataset, worker_init_fn=worker_init_fn, **self.args)
        self.DataLoader_iter = iter(self.DataLoader)   

    def __next__(self):
        if self.DataLoader == None:
            self.build()
        try:
            batch = next(self.DataLoader_iter)
            self.iteration += 1
            return batch
        
        except StopIteration:
            self.epoch += 1
            if self.args['sampler']:
                self.args['sampler'].set_epoch(self.epoch)
            self.iteration = 0
            self.DataLoader_iter = iter(self.DataLoader)
            batch = next(self.DataLoader_iter)
            return batch
    next = __next__

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.dataset)
