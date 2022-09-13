import torch
import torch.nn as nn
import torch.nn.functional as F
from models.freezed_bn import FrozenBatchNorm2d
from models.backbone import build_backbone


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        out_chanels = 256
        ASPPModule = _ASPPModule

        self.aspp1 = ASPPModule(inplanes, out_chanels, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = ASPPModule(inplanes, out_chanels, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = ASPPModule(inplanes, out_chanels, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = ASPPModule(inplanes, out_chanels, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, out_chanels, 1, stride=1, bias=False),
                                             BatchNorm(out_chanels),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(out_chanels*5, out_chanels, 1, bias=False)
        self.bn1 = (BatchNorm(out_chanels))
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x 

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(Decoder, self).__init__()

        low_level_inplanes = 256
        out_chanels = 256
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
                                        nn.Conv2d(out_chanels + 48, out_chanels, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(out_chanels),
                                        nn.ReLU(),
                                        nn.Conv2d(out_chanels, out_chanels, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(out_chanels),
                                        nn.ReLU(),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(out_chanels, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        mid_feat = x = torch.cat((x, low_level_feat), dim=1)

        for i in range(len(self.last_conv)-1):
            x = self.last_conv[i](x)
        out = self.last_conv[-1](x)
        return x, mid_feat, out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, BatchNorm):
    return Decoder(num_classes, BatchNorm)

def build_aspp(output_stride, BatchNorm):
    return ASPP(output_stride, BatchNorm)

class DeepLabV3_plus(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                    bn_backbone='bn', bn_aspp='bn', bn_decoder='bn', use_in=False):
        super(DeepLabV3_plus, self).__init__()
        self.best_iou = 0

        bn_dict = {'bn' : nn.BatchNorm2d,
                   'freezed_bn' : FrozenBatchNorm2d,}

        self.backbone = build_backbone(backbone, output_stride, bn_dict[bn_backbone], use_in)
        self.aspp = build_aspp(output_stride, bn_dict[bn_aspp])
        self.decoder = build_decoder(num_classes, bn_dict[bn_decoder])

        
    def _load_pretrained_model(self):
        if hasattr(self.backbone, '_load_pretrained_model'):
            self.backbone._load_pretrained_model()


    def forward(self, input):

        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        cls_feat, mid_feat, out = self.decoder(x, low_level_feat)
        
        return cls_feat, out

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

def get_deeplab_v3_plus(backbone, num_classes, bn_backbone='bn', bn_aspp='bn', bn_decoder='bn', use_in=False):
    return DeepLabV3_plus(
        backbone=backbone,
        output_stride=16,
        num_classes=num_classes,
        bn_backbone=bn_backbone, 
        bn_aspp=bn_aspp, 
        bn_decoder=bn_decoder, 
        use_in=use_in,
    )


