import sys
import os
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")
sys.path.append(_path)
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..", 'utils')
sys.path.append(_path)

from tqdm import tqdm
import argparse
import torch
import numpy as np
from PIL import Image
from torch import nn
from models.adaptation_model import CustomModel
from data import create_dataset

from default.default import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#------------------------------------- color -------------------------------------------
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    
    return new_mask    

def colorize_save(output_pt_tensor, name, save_path, threshold=0.95):
    output_pt_tensor_logic = torch.softmax(output_pt_tensor.cpu(), dim=1)
    ignore_index = (output_pt_tensor_logic.max(dim=1)[0] <= threshold).numpy().squeeze()

    output_np_tensor = output_pt_tensor.cpu().data[0].numpy()
    mask_np_tensor   = output_np_tensor.transpose(1,2,0) 
    mask_np_tensor   = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_np_tensor[ignore_index] = 250
    mask_Img         = Image.fromarray(mask_np_tensor)
    mask_color       = colorize(mask_np_tensor)  

    name = name.split('/')[-1]
    mask_Img.save('%s/%s' % (save_path, name))
    print('%s/%s' % (save_path, name))
    mask_color.save('%s/%s_color.png' % (save_path,name.split('.')[0]))


def main():

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--save_path", type=str, default='inference_pseudolabels_2')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/inference.yml',
        help="Configuration file to use"
    )
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.test.path = args.checkpoint_path

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # create model
    model = CustomModel(cfg, logger=None)
    model.adaptive_load_nets(model.BaseNet, torch.load(cfg.test.path, map_location="cpu")['DeepLabV3_plus']['model_state'])
    model_gen = model.BaseNet
    model_gen.eval()
    
    # load data
    datasets = create_dataset(cfg) 
    target_loader = datasets.target_train_loader
    target_loader_iter = enumerate(target_loader)

    # upsampling layer
    input_size_target = (cfg.data.target.img_cols, cfg.data.target.img_rows)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    for _ in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        target_image = batch['image']
        target_img_name = batch['file_name']
        with torch.no_grad():
            pred_result = []
            for scale in scales:
                tmp_image = torch.nn.functional.interpolate(target_image, scale, mode='bilinear', align_corners=True)
                _, pred_trg = model_gen(tmp_image.cuda())
                pred_trg = interp_target(pred_trg)
                pred_result.append(pred_trg.cpu())
                _, pred_trg_fliped = model_gen(torch.flip(tmp_image.cuda(), dims=[3]))
                pred_trg_fliped = interp_target(pred_trg_fliped)
                pred_result.append(torch.flip(pred_trg_fliped, dims=[3]).cpu())
            colorize_save(sum(pred_result)/len(pred_result), target_img_name[0], save_path, threshold=0)


if __name__ == '__main__':
    scales = [[768,1536], [1024, 2048], [1536,3072]]
    main()
