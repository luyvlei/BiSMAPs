from models.backbone import resnet, resnet_simclr

def build_backbone(backbone, output_stride, BatchNorm, use_in=False):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, use_in=use_in)
    elif backbone == 'resnet152simclr':
        return resnet_simclr.get_resnet(depth=152, sk_ratio=0.0625)[0]
    else:
        raise NotImplementedError
