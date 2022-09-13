import logging
import os
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
    

def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr = logging.FileHandler(file_path)
    hdlr.setFormatter(formatter)
    hdlr.setLevel(logging.INFO)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

    return logger

def idx_2_color(label, ignore=250):
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
        # ------
        [64, 64, 128],
        [144, 35, 232],
        [170, 70, 70],
        [172, 102, 156],
        [90, 113, 153],
        [53, 113, 153],
        [50, 170, 30],
        [20, 250, 10],
        [7, 242, 35],
        [2, 151, 152],
        [0, 130, 180],
    ]
    colors = np.array(colors)
    color_map = np.zeros((label.shape[0],label.shape[1],3))
    for i in range(colors.shape[0]):
        color_map[label==i] = colors[i]
    return color_map

def idx_save(label, file="tmp.png"):
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if len(label.shape) == 3:
        label = label[0,...]
    plt.imshow(idx_2_color(label)/255.)
    plt.savefig(file)

def validate(valid_loader, device, model, running_metrics_val, scales):
    for batch in tqdm(valid_loader):
        images_val = batch['image'].to(device)
        labels_val = batch['label'].to(device)
        pred_result = []
        for scale in scales:
            tmp_images = F.interpolate(images_val, scale, mode='bilinear', align_corners=True) # images_val# 
            _, outs = model.forward(tmp_images)
            logits = F.softmax(outs, dim=1)
            logits = F.interpolate(logits, labels_val.size()[1:], mode='bilinear', align_corners=True)
            pred_result.append(logits.cpu())
        result = sum(pred_result)
        label_pred = result.max(dim=1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, label_pred)
