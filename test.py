import os
import sys
import shutil
import torch
import random
import argparse
import numpy as np
import torch

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from data import create_dataset
from utils.utils import get_logger,validate
from models.adaptation_model import CustomModel
from metrics import runningScore
from tensorboardX import SummaryWriter
from default.default import cfg


def test(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    ## create dataset
    device = torch.device("cuda:{}".format(0))
    datasets = create_dataset(cfg)  

    model = CustomModel(cfg, logger)
    running_metrics_val = runningScore(cfg.data.target.n_class)
    source_running_metrics_val = runningScore(cfg.data.target.n_class)
    path = cfg.test.path
    checkpoint = torch.load(path, map_location='cpu')
    if path[-3:] == "pkl":
        model.adaptive_load_nets(model.BaseNet, checkpoint['DeepLabV3_plus']['model_state'])
    elif path[-3:] == "pth":
        model.adaptive_load_nets(model.BaseNet, checkpoint)
    else:
        pass
    
    scales = cfg.test.resize_size

    validation(model, logger, writer, datasets, device, running_metrics_val, source_running_metrics_val, iters = model.iter, scales=scales)

def validation(model, logger, writer, datasets, device, running_metrics_val, source_running_metrics_val, iters, scales):
    iters = iters
    model.eval()
    with torch.no_grad():
        validate(datasets.target_valid_loader, device, model, running_metrics_val, scales)
        
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        print('{}: {}'.format(k, v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/test_model.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)

    test(cfg, writer, logger)
