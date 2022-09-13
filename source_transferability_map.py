import os
import torch
import argparse
from tqdm import tqdm
from default.default import cfg
from data import create_dataset
from models.backbone import build_backbone
from sklearn.cluster import KMeans
import joblib
from glob import glob
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
import random

def extract_feat(args):
    if not os.path.exists(args.source_feat_path):
        os.makedirs(args.source_feat_path)
    if not os.path.exists(args.target_feat_path):
        os.makedirs(args.target_feat_path)
    datasets = create_dataset(cfg) 
    source_length = len(datasets.source_train)
    target_length = len(datasets.target_train)
    device = torch.device("cuda", args.gpu)
    feat_extractor = build_backbone("resnet152simclr", output_stride=32, BatchNorm=torch.nn.BatchNorm2d, use_in=False).to(device)
    feat_extractor.load_state_dict(torch.load("./pretrained/r152_1x_sk1.pth", map_location='cpu')['resnet'])
    feat_extractor.eval()
    
    avg_pooling = torch.nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))

    print("loading source features")
    for _ in tqdm(range(source_length)):
        batch = datasets.source_train_loader.next()
        source_image, source_label, source_img_name = batch['image'].to(device), batch['label'].to(device), batch['file_name'][0].split("/")[-1]
        with torch.no_grad():
            feat = feat_extractor(source_image)
            feat = avg_pooling(feat)
        torch.save(feat, os.path.join(args.source_feat_path, source_img_name.split(".")[0] + ".bin"))
    
    print("loading target features")
    for _ in tqdm(range(target_length)):
        batch = datasets.target_train_loader.next()
        target_image, target_label, target_img_name = batch['image'].to(device), batch['label'].to(device), batch['file_name'][0].split("/")[-1]
        with torch.no_grad():
            feat = feat_extractor(target_image)
            feat = avg_pooling(feat)
        torch.save(feat, os.path.join(args.target_feat_path, target_img_name.split(".")[0] + ".bin"))
    

def feat_cluster(args):
    feat_files = glob(args.target_feat_path + "/*.bin")
    feat_list = []
    for f in tqdm(feat_files):
        feat = torch.load(f, map_location="cpu").permute((0,2,3,1)).reshape(-1, 2048)
        feat_list.append(feat)
    
    X = torch.cat(feat_list).numpy()
    samples = np.arange(0,X.shape[0])
    np.random.shuffle(samples)
    print("samples counts : {}".format(X.shape[0]))
    X = X[samples[:args.samples], :]
    kmeans = KMeans(n_clusters=40, random_state=5, verbose=2, n_init=1).fit(X)
    joblib.dump(kmeans, args.kmeans_path)

@torch.no_grad()
def generate_reweighting_map(args):

    skip_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
    class_map = dict(zip(valid_classes, range(len(valid_classes))))  
    def encode_segmap(lbl):
        for _i in skip_classes:
            lbl[lbl == _i] = 250
        for _i in valid_classes:
            lbl[lbl == _i] = class_map[_i]
        return lbl
    
    def encode_segmap_synthia(mask):
        id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        label_copy = 250 * np.ones(mask.shape, dtype=np.uint8)
        for k, v in id_to_trainid.items():
            label_copy[mask == k] = v
        return label_copy

    kmenas = joblib.load(args.kmeans_path)
    centers = torch.Tensor(kmenas.cluster_centers_)[..., None, None]
    
    source_features = glob(args.source_feat_path + "/*.bin")
    mean_distance = None
    ema_momentum = 0.99
    log2 = torch.log(torch.Tensor([2]))
    for f in tqdm(source_features):
        name = f.split("/")[-1].split(".")[0]
        lbl_file = os.path.join(args.source_label_path, name+".png")
        label = Image.open(lbl_file) if not args.synthia else Image.fromarray(cv2.imread(lbl_file,-1)[:,:,-1])
        label = label.resize((args.shape_w, args.shape_h), Image.NEAREST)
        label = encode_segmap(np.array(label)) if not args.synthia else encode_segmap_synthia(np.array(label))

        feat = torch.load(f, map_location="cpu")
        feat_distence = torch.norm(feat-centers, p=2, dim=1) 
        feat_min_distance, _ = feat_distence.min(dim=0) 
        if mean_distance is None: 
            mean_distance = feat_min_distance.mean() 
        else:
            mean_distance = mean_distance * ema_momentum + (1 - ema_momentum) * feat_min_distance.mean()
        weight_map = torch.exp(-(feat_min_distance)**2 / (mean_distance**2/log2))
        weight_mapUp = F.interpolate(weight_map[None,None,...], size=(args.shape_h, args.shape_w), mode="nearest").squeeze().numpy()

        for idx,e in enumerate(norm_class_entropy):
            weight_mapUp[label==idx] = weight_mapUp[label==idx] + e
        weight_mapUp = np.clip(weight_mapUp, 0.0, 1.0)
        Image.fromarray(weight_mapUp*255).convert('L').save(args.reweight_map_path + "/{}.png".format(name))


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

    if not (os.path.exists(args.source_feat_path) and os.path.exists(args.target_feat_path)) or \
        not os.listdir(args.source_feat_path) or not os.listdir(args.target_feat_path):
        extract_feat(args)
    print("feature extract done")

    if not os.path.exists(args.kmeans_path):
        feat_cluster(args)
    print("kmeans cluster done")

    if not os.path.exists(args.reweight_map_path) or not os.listdir(args.reweight_map_path):
        os.mkdir(args.reweight_map_path)
        generate_reweighting_map(args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default="./configs/gta5_stm.yml")
    parser.add_argument("--source_label_path", type=str, default="./dataset/GTA5/labels/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source_feat_path", type=str, default="./resnet152_feats_gta/source")
    parser.add_argument("--target_feat_path", type=str, default="./resnet152_feats_gta/target")
    parser.add_argument("--kmeans_path", type=str, default="./kmeans_r152_gta.model")
    parser.add_argument("--reweight_map_path", type=str, default="./gta5_stm/")
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--shape_w", type=int, default=1914)
    parser.add_argument("--shape_h", type=int, default=1052)
    parser.add_argument("--synthia", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    cfg.merge_from_file(args.config)

    gtabased_class_entropy = [0.1502,0.2935,0.1362,0.4436,0.4393,0.2795,0.3139,0.3106,0.1068,0.3195,0.0719,0.2364,0.4754,0.0986,0.3673,0.4127,0.4826,0.5645,0.6694,]
    synthiabased_class_entropy = [0.1039,0.1333,0.0960,0.4322,0.4646,0.2801,0.3994,0.3480,0.1437,0.0000,0.1032,0.2556,0.4076,0.1399,0.0000,0.3501,0.0000,0.5117,0.4416]
    class_entropy = gtabased_class_entropy if not args.synthia else synthiabased_class_entropy

    min_entropy = min(class_entropy)
    max_entropy = max(class_entropy)
    norm_class_entropy = [(i-min_entropy)/(max_entropy-min_entropy) for i in class_entropy]
    print(norm_class_entropy)
    seed_all(1)
    main(args)
