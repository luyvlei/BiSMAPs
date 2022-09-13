import torch
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, pixel_rewieght=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        raise NotImplementedError('sizes of input and label are not consistent')

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    if pixel_rewieght is None:
        loss = F.cross_entropy(
            input, target, weight=weight, ignore_index=250, size_average=size_average, reduce=True
        )
        return loss
    else:
        pixel_rewieght = pixel_rewieght.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, ignore_index=250, size_average=False, reduce=False
        )
        loss *= pixel_rewieght
        if size_average:
            loss = loss.sum() / target[target!=250].shape[0]
        return loss
   


def sceloss(input, target, aplha=0.1, beta=1.0, n_classes=19):
    """
    logits:     N * C * H * W 
    weight:     N * H * W
    """
    ce = F.cross_entropy(input, target.clone(), ignore_index=250)

    pred = F.softmax(input, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    mask = (target != 250).float()
    target = target.clone()
    target[target==250] = n_classes
    label_one_hot = torch.nn.functional.one_hot(target, n_classes + 1).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
    rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
    return ce * aplha + rce * beta




