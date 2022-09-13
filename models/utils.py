import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GradReverse(nn.Module):
    def forward(self, x):
        return x * 1

    def backward(self, grad_output):
        return (-1 * grad_output)


class NormalisationPoolingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, scale_factor):
        # scale_factor = (softmax_ > 0).sum()
        ctx.scale_factor = scale_factor
        # x = input_ * softmax_
        # pass
        return input_ * 1

    @staticmethod
    def backward(ctx, grad_output):
        scale_factor = ctx.scale_factor
        # print('scale factor: ', scale_factor)
        # print('mean_grad_before scale: ', torch.mean(grad_output))
        grad_input = grad_output * scale_factor
        # print('mean_grad: ', torch.mean(grad_input))
        # input()
        return grad_input, None

class normalisation_pooling(nn.Module):
    def __init__(self, ):
        super(normalisation_pooling, self).__init__()

    def forward(self, input, scale_factor):
        return NormalisationPoolingFunction.apply(input, scale_factor)

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock).__init__()
        pass
    