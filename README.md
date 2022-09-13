# Bidirectional Self-Training with Multiple Anisotropic Prototypes for Domain Adaptive Semantic Segmentation
 (ACM MM 2022)
This is a [pytorch](http://pytorch.org/) implementation of [BiSMAP](Bidirectional Self-Training with Multiple Anisotropic Prototypes for Domain Adaptive Semantic Segmentation).


### Prerequisites
- Python 3.7.0
- GPU Memory >= 11G (Preferably using a gpu with tensor core)
- Pytorch 1.10.0

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download-2/ )

- Download [The Stylized Datasets]( https://mega.nz/folder/JRRXhChK#PzTBXqg5TPKdtf8WIkTuLw ) 

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The pretrained model]( https://mega.nz/folder/FQgUnTxZ#2aJQMP5zRuL7Vbu-ZtyqrA )

The data folder is structured as follows:
```
├── dataset/
│   ├── CityScape/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
│   ├── gta_stylized/
│   └── 			
└── pretrained/
│   ├── r101_1x_sk0.pth
│   ├── r152_1x_sk1.pth
│   ├── gta5_dill_model.pkl
│   ├── gta5_warmup_model.pkl
...
```

### Evaluate models
Our final model is available via [MEGA]( https://mega.nz/folder/EU4mRRxD#qHFPKcgOx7x-hwcE7K6D8w )
```
python3 test.py --config ./configs/gta5_test.yml 
```
---
### Train models
#### Generate Source Transferability Map and Pseudo Labels
```
python source_transferability_map.py --config ./configs/gta5_stm.yml
```
```
python generate_maps_pseudolabel.py --config configs/gta5_maps_pla.yml --threshold 100
```

#### Self-Training
```
python train.py --config configs/gta5_st.yml --logdir ./runs/gta5_st
```

#### Self-Distillation (two stage)
```
# distill 1
python3 inference.py --config ./configs/inference.yml --checkpoint_path ./runs/gta5_st/from_gta5_to_cityscapes_on_deeplabv3_plus_best_model.pkl --save_path pseudolabels_dill
python3 train.py --gpu 1 --config ./configs/gta5_dill.yml --dill_teacher ./runs/gta5_st/from_gta5_to_cityscapes_on_deeplabv3_plus_best_model.pkl --logdir ./runs/gta5_dill_1
```
```
# distill 2
python3 inference.py --config ./configs/inference.yml --checkpoint_path ./runs/gta5_dill_1/from_gta5_to_cityscapes_on_deeplabv3_plus_best_model.pkl --save_path pseudolabels_dill
python3 train.py --gpu 1 --config ./configs/gta5_dill.yml --dill_teacher ./runs/gta5_dill_1/from_gta5_to_cityscapes_on_deeplabv3_plus_best_model.pkl --logdir ./runs/gta5_dill_2
```


### Citation
If you use this code in your research please consider citing
```
@article{lu2022bidirectional,
  title={Bidirectional Self-Training with Multiple Anisotropic Prototypes for Domain Adaptive Semantic Segmentation},
  author={Lu, Yulei and Luo, Yawei and Zhang, Li and Li, Zheyang and Yang, Yi and Xiao, Jun},
  journal={arXiv preprint arXiv:2204.07730},
  year={2022}
}
```