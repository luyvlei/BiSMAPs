import os
import sys
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import logging

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from data import create_dataset
from utils.utils import get_logger, validate
from models.adaptation_model import CustomModel
from metrics import runningScore, averageMeter
from tensorboardX import SummaryWriter
from default.default import cfg

def train(gpu, args):

    # when use ddp, the gpu relate to the current worker id
    cfg.training.device = gpu
    if args.ddp:
        cfg.training.device = cfg.training.ddp.gpus[gpu]
    
    writer, logger = None, None
    if not args.ddp or (args.ddp and gpu==0):
        if args.logdir is None:
            run_id = random.randint(1, 100000)
            today = time.strftime("%Y_%m_%d_%H_%M_",time.localtime(time.time()))
            logdir = os.path.join('runs', os.path.basename(args.config)[:-4], today + str(run_id))
        else:
            logdir = args.logdir
        writer = SummaryWriter(log_dir=logdir)
        shutil.copy(args.config, logdir)
        logger = get_logger(logdir)
    
    # 很奇怪
    logging.info("begin")

    seed_all(cfg.get('seed', 100))

    if args.ddp:
        torch.distributed.init_process_group(backend="nccl", world_size=len(cfg.training.ddp.gpus), rank=gpu)
    torch.cuda.set_device(cfg.training.device)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_frac, gpu)
    
    # create dataset
    device = torch.device("cuda:{}".format(cfg.training.device))
    datasets = create_dataset(cfg) 

    # create model
    model = CustomModel(cfg, logger)

    # Setup Metrics
    running_metrics_val = runningScore(cfg.data.target.n_class)
    source_running_metrics_val = runningScore(cfg.data.target.n_class)
    time_meter = averageMeter() # record time cost of a step
    dataloader_time_meter = averageMeter() # record time cost of dataloader

    # multi_scale_eval
    scales = cfg.test.resize_size

    # samples per step
    samples_per_step = cfg.data.source.batch_size * (model.source_forward_flag) \
                    + cfg.data.target.batch_size * (model.target_forward_flag)

    # resmue
    start_iter = 0
    if cfg.training.iter_resume and cfg.training.resume_flag:
        start_iter = model.iter

    # begin training
    for iter in range(start_iter, cfg.training.train_iters):
        
        start_ts = time.time()
        model.iter = iter

        target_batch = None
        if cfg.training.loss_target_seg or cfg.training.loss_consist:
            target_batch = datasets.target_train_loader.next()

        source_batch = None
        if cfg.training.loss_source_seg:
            source_batch = datasets.source_train_loader.next()

        dataloader_time_meter.update(time.time() - start_ts)

        model.train()
        loss, loss_source, loss_consist, loss_target, loss_dill = model.step(source_batch, target_batch)
        model.optimizer_step()
        model.scheduler_step()

        time_meter.update(time.time() - start_ts)
        if (iter + 1) % cfg.training.print_interval == 0 and (not args.ddp or (args.ddp and gpu==0)):
            fmt_str = 'Iter [{:d}/{:d}] lr: {:.6f} head_lr: {:.6f} Loss: {:.4f} Loss_source: {:.4f}  Loss consist: {:.4f} Loss tgt: {:.4f} Loss dill {:.4f} Time_step/Image: {:.3f} Time_data/Image: {:.3f}'
            print_str = fmt_str.format(
                                    iter + 1,
                                    cfg.training.train_iters, 
                                    model.BaseOpti.state_dict()['param_groups'][0]['lr'],
                                    model.BaseOpti.state_dict()['param_groups'][1]['lr'] if len(model.BaseOpti.state_dict()['param_groups']) > 1 else 0,
                                    loss,
                                    loss_source,
                                    loss_consist,
                                    loss_target,
                                    loss_dill,
                                    time_meter.avg / samples_per_step,
                                    dataloader_time_meter.avg / samples_per_step)
            logger.info(print_str)
            writer.add_scalar('loss/train_loss', loss, iter+1)
            writer.add_scalar('loss/train_loss_source', loss_source, iter+1)
            writer.add_scalar('loss/train_loss_consist', loss_consist, iter+1)
            writer.add_scalar('loss/train_loss_target', loss_target, iter+1)
            time_meter.reset()

            # evaluation
            if (iter + 1) % cfg.training.val_interval == 0 or (iter + 1) == cfg.training.train_iters:
                validation(
                    model, logger, writer, datasets, device, running_metrics_val, \
                    source_running_metrics_val, iters = model.iter, scales=scales
                    )
                logger.info('Best iou until now is {}'.format(model.best_iou))


def validation(model, logger, writer, datasets, device, running_metrics_val,\
        source_running_metrics_val, iters, scales):

    model.eval()
    with torch.no_grad():
        validate(datasets.target_valid_loader, device, model, running_metrics_val, scales)
        
    score, class_iou = running_metrics_val.get_scores()
    class_names = [
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
    for k, v in score.items():
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/{}'.format(k), v, iters+1)

    for idx, (k, v) in enumerate(class_iou.items()):
        logger.info('{}: {:.4f} {}'.format(k, v, class_names[idx]))
        writer.add_scalar('val_metrics/cls_{}'.format(k), v, iters+1)

    running_metrics_val.reset()
    source_running_metrics_val.reset()

    state = {}
    new_state = {
        "model_state": model.BaseNet.state_dict(),
        "optimizer_state": model.BaseOpti.state_dict(),
        "scheduler_state": model.BaseSchedule.state_dict(),                            
    }
    state[model.BaseNet.__class__.__name__] = new_state
    state['iter'] = iters + 1
    state['best_iou'] = score["Mean IoU : \t"]
    state['configs'] = dict(model.cfg)
    save_path = os.path.join(writer.file_writer.get_logdir(),
                                "from_{}_to_{}_on_{}_current_model.pkl".format(
                                    cfg.data.source.name,
                                    cfg.data.target.name,
                                    cfg.model.arch))
    torch.save(state, save_path)

    if score["Mean IoU : \t"] >= model.best_iou:
        model.best_iou = score["Mean IoU : \t"]
        state = {}
        new_state = {
            "model_state": model.BaseNet.state_dict(),
            "optimizer_state": model.BaseOpti.state_dict(),
            "scheduler_state": model.BaseSchedule.state_dict(),
        }
        state[model.BaseNet.__class__.__name__] = new_state
        state['iter'] = iters + 1
        state['best_iou'] = model.best_iou
        state['configs'] = dict(model.cfg)
        save_path = os.path.join(writer.file_writer.get_logdir(),
                                    "from_{}_to_{}_on_{}_best_model.pkl".format(
                                        cfg.data.source.name,
                                        cfg.data.target.name,
                                        cfg.model.arch))
        torch.save(state, save_path)
    return score["Mean IoU : \t"]

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(1337)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '3335'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default='./configs/warmup_gta5_stm.yml')
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_memory_frac", type=float, default=1.0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp_gpus", type=str, default=None)
    parser.add_argument("--dill_teacher", type=str, default=None)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    if args.dill_teacher is not None:
        cfg.training.distillation_resume = args.dill_teacher

    cfg.training.ddp.status = args.ddp
    if args.ddp_gpus:
        cfg.training.ddp.gpus = [int(i) for i in args.ddp_gpus.split(",")]
    else:
        cfg.training.ddp.gpus = [i for i in range(torch.cuda.device_count())]
    
    if args.ddp:
        mp.spawn(train, nprocs=len(cfg.training.ddp.gpus), args=(args))
    else:
        train(args.gpu, args)
