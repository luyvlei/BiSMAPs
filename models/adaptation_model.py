import os
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from schedulers import get_scheduler
from models.deeplabv3_plus import get_deeplab_v3_plus
from loss import get_loss_function
from loss.loss import sceloss
from utils.utils import get_logger
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from default.default import cfg

class CustomModel():
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger if logger is not None else get_logger("./tmp/")
        self.best_iou = -100
        self.iter = 0
        self.device = torch.device("cuda:{}".format(cfg.training.device))
        self.scaler = GradScaler() if cfg.training.scaler else None
        self.optimizers = []

        # constract the segmentation model
        arch = {
            'deeplabv3_plus' : get_deeplab_v3_plus,
        }[cfg.model.arch]
        arch_params = {
            'backbone' : cfg.model.basenet.version,
            'num_classes' : cfg.data.source.n_class,
            'bn_backbone' : cfg.model.bn_backbone, 
            'bn_aspp' : cfg.model.bn_aspp, 
            'bn_decoder' : cfg.model.bn_decoder, 
            'use_in' : False
        }
        self.BaseNet = arch(**arch_params).to(self.device)
        self.logger.info('The model is {} with backbone'.format(cfg.model.arch, cfg.model.basenet.version))

        # constract the optimizer for segmentation model
        optim_map = {
            'SGD':torch.optim.SGD,
            'Adam':torch.optim.Adam,
        }
        optimizer_cls = optim_map[cfg.training.optimizer.name]
        optim_args = {k:v for k,v in cfg.training.optimizer.items() if k in inspect.getfullargspec(optimizer_cls).args and k != "lr"}
        lr = cfg.training.optimizer.lr
        param = [
        {'params': self.BaseNet.get_1x_lr_params(), "lr": lr},
        {'params': self.BaseNet.get_10x_lr_params(), "lr": lr*10 if cfg.training.head_10x_lr else lr},
        ]
        self.BaseOpti = optimizer_cls(param, **optim_args) if cfg.model.arch != 'segformer' else optimizer_cls(self.BaseNet.parameters(), lr=lr, **optim_args)
        self.BaseSchedule = get_scheduler(self.BaseOpti, cfg.training.lr_schedule)

        self.BaseNetDDP = self.init_device(self.BaseNet, self.device, whether_DP=cfg.training.ddp.status)
        self.optimizers.append(self.BaseOpti)

        # 1.init weights 2.load pretrained backbone 3.load checkpoints if needed
        self.setup(cfg)

        # ema model
        self.BaseNet_ema = arch(**arch_params).to(self.device)
        for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
            param_k.data = param_q.data.clone()
        for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.BaseNet_ema.buffers()):
            buffer_k.data = buffer_q.data.clone()
        for param in self.BaseNet_ema.parameters():
            param_k.requires_grad = False
        self.BaseNet_ema.train()

        # flags for adversarial training
        self.adv_source_label = 0
        self.adv_target_label = 1

        # loss function for training
        self.loss_source = get_loss_function(cfg)
        self.loss_target = get_loss_function(cfg)
        self.loss_sce = sceloss

        # distillation teacher
        if self.cfg.training.distillation:
            self.distill_teacher = arch(**arch_params).to(self.device)
            teacher_checkpoint = torch.load(self.cfg.training.distillation_resume)
            self.adaptive_load_nets(self.distill_teacher, teacher_checkpoint["DeepLabV3_plus"]['model_state'])
            self.distill_teacher.eval()
            self.adaptive_load_nets(self.BaseNet.backbone, torch.load('pretrained/r101_1x_sk0.pth')['state_dict'])

        # forward flag
        self.source_forward_flag = self.cfg.training.loss_source_seg
        self.target_forward_flag = self.cfg.training.loss_target_seg
        

    def setup(self, cfg):
        '''
        set optimizer and load pretrained model
        '''
        net = self.BaseNet
        self.init_weights(cfg.model.init, net)

        if hasattr(net, '_load_pretrained_model') and cfg.model.pretrained:
            print("loading pretrained model for {}".format(net.__class__.__name__))
            net._load_pretrained_model()

        '''load pretrained model'''
        if cfg.training.resume_flag:
            self.load_nets(cfg)

    def ema_update(self):
        for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
                param_k.data = param_k.data.clone() * 0.999 + param_q.data.clone() * (1. - 0.999)
        for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.BaseNet_ema.buffers()):
                buffer_k.data = buffer_q.data.clone()


    def forward(self, input, upsample=False):
        feat_cls, output = self.BaseNetDDP(input)
        if upsample and output.size() != input.size():
            output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        return feat_cls, output
    
    def get_threshold(self, predicted_prob, predicted_label, valid_alpha=0.2):
        thresholds = np.zeros(19)
        for c in range(19):
            x = predicted_prob[predicted_label==c]
            if len(x) == 0:
                continue        
            x = np.sort(x)
            x = x[::-1]
            thresholds[c] = max(x[np.int(np.floor(len(x)*(valid_alpha)))-1], 0)

        return thresholds

    def step(self, source_batch, target_batch):
        # data init
        source_x = source_label = reweight_map = None
        target_x = target_label = None
        if source_batch:
            source_x = source_batch['image'].to(self.device)
            source_label = source_batch['label'].to(self.device)
            reweight_map = None if not cfg.data.source.use_reweight_map else source_batch['reweight_map'].to(self.device)
        if target_batch:
            target_x = target_batch['image'].to(self.device)
            target_label = target_batch['label'].to(self.device)
        # losses init
        loss = torch.Tensor([0]).to(self.device).item()
        loss_source = torch.Tensor([0]).to(self.device)
        loss_target = torch.Tensor([0]).to(self.device)
        loss_consist = torch.Tensor([0]).to(self.device)

        if self.source_forward_flag:
            with autocast(enabled=self.cfg.training.scaler):
                _, source_outputUp = self.forward(input=source_x, upsample=True)
                loss_source += self.loss_source(input=source_outputUp, target=source_label, pixel_rewieght=reweight_map)
            self.amp_backward(loss_source, self.BaseOpti)
            loss += loss_source.item()

        if self.cfg.training.loss_consist:
            self.ema_update()
            target_x_masked = target_batch['masked_img'].to(self.device)
            mask = target_batch['mask'].to(self.device)

            with autocast(enabled=self.cfg.training.scaler):
                # ema forward
                with torch.no_grad():
                    self.BaseNet_ema.train()
                    self.BaseNet_ema.decoder.last_conv[-2].eval()
                    _, ema_out = self.BaseNet_ema(target_x)

                # base forward
                _, base_out = self.BaseNet(target_x_masked)

                ema_outUp = F.interpolate(ema_out, size=target_x.size()[2:], mode='bilinear', align_corners=True)
                base_outUp = F.interpolate(base_out, size=target_x.size()[2:], mode='bilinear', align_corners=True)

                ema_outUp_softmax = torch.softmax(ema_outUp, dim=1).permute((0,2,3,1))
                base_outUp_softmax = torch.log_softmax(base_outUp, dim=1).permute((0,2,3,1))
                loss_consist = self.cfg.training.lambda_consist * self.get_consist_weight(start_it=0,rampup_length=4000) * nn.KLDivLoss()(base_outUp_softmax[mask==False], ema_outUp_softmax[mask==False])

            self.amp_backward(loss_consist, self.BaseOpti)
            loss += loss_consist.item()
        
        if self.target_forward_flag:
            with autocast(enabled=self.cfg.training.scaler):
                _, target_outputUp = self.forward(target_x, upsample=True)
        
        # sup loss
        if self.cfg.training.loss_target_seg:
            with autocast(enabled=self.cfg.training.scaler):
                loss_target_sup = (self.loss_target(target_outputUp, target_label) if not self.cfg.training.sce_loss_target else self.loss_sce(target_outputUp, target_label, n_classes=19))
                loss_target += loss_target_sup
        
        loss_kd = torch.Tensor([0])
        if self.cfg.training.distillation:
            with autocast(enabled=self.cfg.training.scaler):
                student = F.softmax(target_outputUp, dim=1)
                with torch.no_grad():
                    _, teacher_out = self.distill_teacher(target_x)
                    teacher_out = F.interpolate(teacher_out, size=target_outputUp.shape[2:], mode='bilinear', align_corners=True)
                    teacher = F.softmax(teacher_out, dim=1)
                loss_kd = F.kl_div(student, teacher, reduction='none')
                mask = (teacher != 250).float()
                loss_kd = (loss_kd * mask).sum() / mask.sum()
                loss_target += loss_kd

            
        if self.target_forward_flag:    
            self.amp_backward(loss_target, self.BaseOpti)
            loss += loss_target.item()

        return loss, loss_source.item(), loss_consist.item(), loss_target.item(), loss_kd.item()


    def scheduler_step(self):
        self.BaseSchedule.step()
    
    
    def optimizer_step(self):
        if self.cfg.training.scaler:
            for optim in self.optimizers:
                self.scaler.step(optim)
                optim.zero_grad()
            self.scaler.update()
        else:
            for optim in self.optimizers:
                optim.step()
                optim.zero_grad()


    def init_device(self, net, device, whether_DP=False):
        net = net.to(device)
        if whether_DP:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.cfg.training.device], find_unused_parameters=True)
        return net

    def amp_backward(self, loss, optim, retain_graph=False):
        if self.cfg.training.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward(retain_graph=retain_graph)

    
    def eval(self, net=None):
        """Make specific models eval mode during test time"""
        if net == None:
            self.BaseNet.eval()
            self.logger.info("Successfully set the model eval mode") 
        else:
            net.eval()
            self.logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return


    def train(self, net=None):
        if net==None:
            self.BaseNet.train()
        else:
            net.train()
        return
        

    def init_weights(self, cfg, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        init_type = cfg.get('init_type', init_type)
        init_gain = cfg.get('init_gain', init_gain)

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_() 

        net.apply(init_func) 
        

    def adaptive_load_nets(self, net, model_weight):
        """load net/optimizer/scheduler"""
        def rm_module(str):
            if str.split(".")[0] == 'module':
                return ".".join(str.split(".")[1:])
            return str

        self.logger.info("The following item in checkpoint will not merge into model:")
        for k, v in model_weight.items():
            if rm_module(k) not in net.state_dict():
                self.logger.info(rm_module(k))
        model_dict = net.state_dict()
        pretrained_dict = {rm_module(k) : v for k, v in model_weight.items() if rm_module(k) in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    def get_consist_weight(self, start_it, rampup_length):
        if rampup_length == 0 and start_it==0:
            return 1.0
        else:
            current = np.clip(self.iter-start_it, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase)) 

    def load_nets(self, cfg): # load pretrained weights on the net
        if not os.path.isfile(cfg.training.resume):
            raise Exception("No checkpoint found at '{}'".format(cfg.training.resume))

        if cfg.training.resume[-3:] == "pkl": # pkl包含了所有参数，包括优化器
            
            self.logger.info("Loading model and optimizer from pkl file '{}'".format(cfg.training.resume))
            self.checkpoint = checkpoint = torch.load(cfg.training.resume, map_location='cpu')

            name = self.BaseNet.__class__.__name__
            if checkpoint.get(name) == None:
                return 

            self.adaptive_load_nets(self.BaseNet, checkpoint[name]["model_state"])
            if cfg.training.optimizer_resume:
                self.adaptive_load_nets(self.BaseOpti, checkpoint[name]["optimizer_state"])
                self.adaptive_load_nets(self.BaseSchedule, checkpoint[name]["scheduler_state"])

            self.iter = checkpoint["iter"]
            self.best_iou = checkpoint['best_iou']
            self.logger.info("Loaded checkpoint '{}' (iter {})".format(cfg.training.resume, checkpoint["iter"]))

        elif cfg.training.resume[-3:] == "pth": # pth只有模型参数
            self.logger.info("Loading model and optimizer from pth file '{}'".format(cfg.training.resume))
            checkpoint = torch.load(cfg.training.resume, map_location='cpu')
            self.adaptive_load_nets(self.BaseNet, checkpoint)
            self.logger.info("Loaded checkpoint '{}' ".format(cfg.training.resume))


