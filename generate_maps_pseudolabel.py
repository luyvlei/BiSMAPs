import os
import sys
from cv2 import threshold
import torch
import argparse
import numpy as np
import torch
import time
import math
import random
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from gmm_torch.gmm import GaussianMixture as GMM,load,dump

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from data import create_dataset
from models.adaptation_model import CustomModel
from default.default import cfg
from torch.cuda.amp import autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"


class Feature_saver:
    def __init__(self, class_ids, save_path, pool_size=100000):
        self.features = {i:None for i in class_ids}
        self.feature_pool_count = {i:0 for i in class_ids}
        self.feat_file_count = {i:0 for i in class_ids}
        self.pool_size = pool_size
        self.class_ids = class_ids
        self.save_pathes = {i:os.path.join(save_path,"feat_{}".format(i)) for i in class_ids}
        for path in self.save_pathes.values():
            if not os.path.exists(path):
                os.mkdir(path)

    def push(self, feat, c_id):
        if feat.shape[0] == 0:
            return
        indexes = torch.randperm(feat.shape[0])[:int(feat.shape[0]*class_weights[c_id])]
        if self.features[c_id] is None:
            self.features[c_id] = [feat[indexes]]
        else:
            self.features[c_id].append(feat[indexes])
        self.feature_pool_count[c_id] += indexes.shape[0]
        if self.feature_pool_count[c_id] > self.pool_size:
            self._save(c_id)

    def _save(self, c_id):
        print("save {}".format(c_id))
        feat = torch.cat(self.features[c_id], dim=0)
        torch.save(feat[:self.pool_size], os.path.join(self.save_pathes[c_id],"{}.bin".format(self.feat_file_count[c_id])))
        self.features[c_id] = None
        self.feature_pool_count[c_id] = 0
        self.feat_file_count[c_id] += 1
    
    def save_left(self):
        for c_id in self.class_ids:
            self._save(c_id)

def save_features(args):

    feat_path = os.path.join(args.save_path, "features")
    os.mkdir(feat_path)

    device = torch.device("cuda:{}".format(cfg.training.device))

    model = CustomModel(cfg)
    model.eval()
    datasets = create_dataset(cfg)

    feat_saver = Feature_saver(class_ids=class_ids, save_path=feat_path, pool_size=1000000)

    for idx in tqdm(range(len(datasets.source_train)//cfg.data.source.batch_size)):
        batch = datasets.source_train_loader.next()
        img = batch['image'].to(device)
        label = batch['label'].to(device)
        with torch.no_grad():
            with autocast():
                feat_cls, output = model.forward(img)
                if args.flip:
                    fliped_img = torch.flip(img, dims=(3,))
                    feat_cls_fliped, _ = model.forward(fliped_img)
                    feat_cls = (feat_cls + torch.flip(feat_cls_fliped, dims=(3,)))/2
        label = F.interpolate(label.unsqueeze(1).double(), size=output.shape[2:], mode="nearest").long().squeeze()
        feat_cls = feat_cls.permute((0, 2, 3, 1)).float()
        pred = torch.argmax(output, dim=1)
        pred = F.interpolate(pred.unsqueeze(1).double(), size=output.shape[2:], mode="nearest").long().squeeze()
        
        for c_id in class_ids:
            feat = feat_cls[(label==c_id) & (pred==label)]
            feat_saver.push(feat.cpu(), c_id)
    
    feat_saver.save_left()
    print("feature save done")

def train_gmm(args):
    os.mkdir(os.path.join(args.save_path, "gmm_models"))
    for c_id in tqdm(class_ids):
        feat = torch.load(os.path.join(args.save_path, "features", "feat_{}".format(c_id), "0.bin")).cuda(args.gpu)
        indexes = torch.randperm(feat.shape[0])[:300000].cuda(args.gpu)
        gmm = GMM(n_components=args.K, n_features=256, covariance_type="diag").cuda(args.gpu).fit(feat[indexes], 1e-3).cpu()
        dump(gmm, os.path.join(args.save_path, "gmm_models")+'/{}_gmm.pth'.format(c_id))
        torch.cuda.empty_cache()

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_save(mask, path):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask.save(path) 


def generate_pseudo_label(args):
    os.mkdir(os.path.join(args.save_path, "pseudo_labels"))

    device = torch.device("cuda:{}".format(cfg.training.device))
    cfg.data.target.batch_size = 1

    model = CustomModel(cfg)
    model.eval()
    datasets = create_dataset(cfg)

    gmms = {}
    for c_id in class_ids:
        gmms[c_id] = load(os.path.join(args.save_path, "gmm_models"+"/{}_gmm.pth".format(c_id)), "diag").cuda(args.gpu)


    for _ in tqdm(range(len(datasets.target_train))):
        batch = datasets.target_train_loader.next()
        img = batch['image'].to(device)
        file_name = batch['file_name']
        with torch.no_grad():
            feat_cls, _ = model.forward(img)
            feat_cls = F.interpolate(feat_cls, size=img.size()[2:], mode='bilinear', align_corners=True)
            if args.flip:
                fliped_img = torch.flip(img, dims=(3,))
                feat_cls_fliped, _ = model.forward(fliped_img)
                feat_cls_fliped = F.interpolate(feat_cls_fliped, size=img.size()[2:], mode='bilinear', align_corners=True)
                feat_cls = (feat_cls + torch.flip(feat_cls_fliped, dims=(3,)))/2
        feat_cls = feat_cls.permute((0, 2, 3, 1)).reshape(-1, 256)
        log_pdf = torch.zeros((feat_cls.shape[0], 19)).to(device) - 99999999

        for c_id in class_ids:
            feat_size, p_size = feat_cls.shape[0], 200000
            for i in range(math.ceil(feat_size / p_size)):
                log_pdf[i*p_size:(i+1)*p_size, c_id] = gmms[c_id].score(feat_cls[i*p_size:(i+1)*p_size,:])
        
        logits, predict = log_pdf.max(dim=1)

        predict[logits<args.threshold] = 250
        pseudo_label = predict.reshape(img.size()[2:]).cpu().numpy()
        colorize_save(pseudo_label, os.path.join(args.save_path, "pseudo_labels",file_name[0].split("/")[-1]))


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

def main(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    if not os.path.exists(os.path.join(args.save_path, "features")):
        save_features(args)
    
    if not os.path.exists(os.path.join(args.save_path, "gmm_models")):
        train_gmm(args)

    if not os.path.exists(os.path.join(args.save_path, "pseudo_labels")):
        generate_pseudo_label(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default='./configs/gta5_maps_pla.yml')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--threshold", type=float, default=100)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--save_path", type=str, default="gta5_maps_workdir")
    parser.add_argument("--synthia", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg.merge_from_file(args.config)
    cfg.training.device = args.gpu
    cfg.training.resume = args.checkpoint_path

    if not args.synthia:
        class_ids = [i for i in range(19)]
        class_weights = [0.002, 0.01, 0.003, 0.1, 0.1, 0.1, 1.0, 1.0, 0.01, 0.05, 0.01, 0.5, 1.0,  0.05, 0.1, 0.2, 1.0, 1.0, 1.0]
    else:
        class_ids = [0,1,2,3,4,5,6,7,8,10,11,12,13,15,17,18]
        class_weights = [0.008, 0.04, 0.012, 0.4, 0.4, 0.4, 1.0, 0.5, 0.04, 0.2, 0.02, 0.07, 0.5,  0.05, 0.4, 0.08, 1.0, 1.0, 1.0]

    seed_all(1)
    main(args)
