from yacs.config import CfgNode as CN
 
_C = CN()

_C.trainset = 'gta5'
_C.valset = 'cityscapes'

# model config
_C.model = CN()
_C.model.use_in = False
_C.model.init = CN()
_C.model.init.init_type = 'kaiming'
_C.model.init.init_gain = 0.02
_C.model.arch = 'deeplabv3_plus'
_C.model.pretrained = True
_C.model.bn_backbone = 'bn'
_C.model.bn_aspp = 'bn'
_C.model.bn_decoder = 'bn'
_C.model.basenet = CN()
_C.model.basenet.version = 'resnet101'


# source train dataset config
_C.data = CN()
_C.data.source = CN()
_C.data.source.name = 'gta5'
_C.data.source.rootpath = 'dataset/GTA5'
_C.data.source.imagepath = None # 默认为空，否则使用该路径下的图片
_C.data.source.split = 'all'
_C.data.source.img_rows = 1052
_C.data.source.img_cols = 1914
_C.data.source.batch_size = 1
_C.data.source.is_transform = True 
_C.data.source.img_norm = True
_C.data.source.mean = [0.485, 0.456, 0.406]
_C.data.source.std = [0.229, 0.224, 0.225]
_C.data.source.use_bgr = False
_C.data.source.shuffle = True
_C.data.source.n_class = 19
_C.data.source.len = None # 数据集长度
_C.data.source.use_reweight_map = False 
_C.data.source.reweight_map_path = ""
_C.data.source.augmentations = CN()
_C.data.source.augmentations.rsize = None
_C.data.source.augmentations.gamma = None
_C.data.source.augmentations.hue = None
_C.data.source.augmentations.brightness = None
_C.data.source.augmentations.saturation = None
_C.data.source.augmentations.contrast = None
_C.data.source.augmentations.rcrop = None
_C.data.source.augmentations.hflip = None
_C.data.source.num_workers = 1

# target train dataset config
_C.data.target = CN()
_C.data.target.name = 'cityscapes'
_C.data.target.rootpath = 'dataset/CityScape'
_C.data.target.labelpath = None # 伪标签路径，默认空则使用gt
_C.data.target.split = 'train'
_C.data.target.img_rows = 1024
_C.data.target.img_cols = 2048
_C.data.target.batch_size = 1
_C.data.target.is_transform = True
_C.data.target.img_norm = True # 是否先归一化
_C.data.target.mean = [0.485, 0.456, 0.406]
_C.data.target.std = [0.229, 0.224, 0.225]
_C.data.target.use_bgr = False
_C.data.target.shuffle = True
_C.data.target.n_class = 19
_C.data.target.augmentations = CN()
_C.data.target.augmentations.rsize = None
_C.data.target.augmentations.gamma = None
_C.data.target.augmentations.hue = None
_C.data.target.augmentations.brightness = None
_C.data.target.augmentations.saturation = None
_C.data.target.augmentations.contrast = None
_C.data.target.augmentations.rcrop = None
_C.data.target.augmentations.hflip = None
_C.data.target.num_workers = 1

# source val dataset config
_C.data.source_valid = CN()
_C.data.source_valid.name = 'gta5'
_C.data.source_valid.rootpath = 'dataset/GTA5'
_C.data.source_valid.imagepath = None # 默认为空，则使用rootpath下的路径
_C.data.source_valid.split = 'val'
_C.data.source_valid.img_rows = 1052
_C.data.source_valid.img_cols = 1914
_C.data.source_valid.batch_size = 1
_C.data.source_valid.is_transform = True 
_C.data.source_valid.img_norm = True
_C.data.source_valid.mean = [0.485, 0.456, 0.406]
_C.data.source_valid.std = [0.229, 0.224, 0.225]
_C.data.source_valid.use_bgr = False
_C.data.source_valid.shuffle = False
_C.data.source_valid.n_class = 19
_C.data.source_valid.len = None
_C.data.source_valid.use_reweight_map = False 
_C.data.source_valid.reweight_map_path = ""
_C.data.source_valid.num_workers = 1

# target val dataset config
_C.data.target_valid = CN()
_C.data.target_valid.name = 'cityscapes'
_C.data.target_valid.rootpath = 'dataset/CityScape'
_C.data.target_valid.labelpath = None # 伪标签路径，默认空则使用gt
_C.data.target_valid.split = 'val'
_C.data.target_valid.img_rows = 1024
_C.data.target_valid.img_cols = 2048
_C.data.target_valid.batch_size = 1
_C.data.target_valid.is_transform = True 
_C.data.target_valid.img_norm = True
_C.data.target_valid.mean = [0.485, 0.456, 0.406]
_C.data.target_valid.std = [0.229, 0.224, 0.225]
_C.data.target_valid.use_bgr = False
_C.data.target_valid.shuffle = False
_C.data.target_valid.n_class = 19
_C.data.target_valid.len = None
_C.data.target_valid.num_workers = 1

_C.data.n_class = 19

# training config
_C.training = CN()
_C.training.ddp = CN()
_C.training.ddp.status = False
_C.training.ddp.gpus = [0]
_C.training.device = 0
_C.training.scaler = False
_C.training.train_iters = 90000
_C.training.val_interval = 1000
_C.training.print_interval = 50
_C.training.loss_source_seg = True
_C.training.sce_loss_target = False
_C.training.loss_target_seg = False
_C.training.loss_consist = False
_C.training.lambda_consist = 20
_C.training.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] 
_C.training.distillation = False
_C.training.distillation_resume = ''

_C.training.optimizer = CN()
_C.training.optimizer.name = 'SGD' # "Adam"
_C.training.optimizer.lr = 1e-3
_C.training.optimizer.weight_decay = 5.0e-4
_C.training.optimizer.momentum = 0.9
_C.training.optimizer.betas = [0.9, 0.999]


# seg loss
_C.training.loss = CN()
_C.training.loss.name = 'cross_entropy'

# 
_C.training.head_10x_lr = False


# lr schedule
_C.training.lr_schedule = CN()
_C.training.lr_schedule.name = 'poly_lr'
_C.training.lr_schedule.gamma = 0.9
_C.training.lr_schedule.eta_min = 1e-5
_C.training.lr_schedule.T_max = 90000
_C.training.lr_schedule.warmup_iters = 400


_C.training.resume = ''
_C.training.resume_flag = False
_C.training.optimizer_resume = False
_C.training.iter_resume = False

# test 
_C.test = CN()
_C.test.path = ''
_C.test.resize_size = [[1024, 2048]]

cfg = _C







