import torch
import numpy as np
import torch.nn as nn
from .ssim3d import MultiScaleSSIMLoss3d as ms_ssim3d
from .ssim3d import SSIMLoss as ssim2d
from .ssim3d import ms_ssim_loss
from torch.nn import functional as F

class ssim3DLoss(nn.Module):
    """Calculate loss function of RGT"""
    def __init__(self):
        super(ssim3DLoss, self).__init__()
        self.ssim = ms_ssim3d()
        self.name = "SSIM3D"
    def forward(self, output, target):
        loss = (1 - self.ssim(output, target))
        return loss
    def getLossName(self):
        return self.name


class dicessim(nn.Module):

    def __init__(self, a, b):
        super(dicessim, self).__init__()
        self.dice = diceloss()
        self.ssim = ssim2DLoss()
        self.name = 'dicessim'
        self.a = a
        self.b = b

    def forward(self, y_true, y_pred):
        d = self.dice(y_true, y_pred)
        c = self.ssim(y_true, y_pred)
        loss = self.a * d + self.b * c
        return loss

    def getLossName(self):
        return self.name


class ssim2DLoss(nn.Module):
    """Calculate loss function of RGT"""
    def __init__(self):
        super(ssim2DLoss, self).__init__()
        self.ssim = ssim2d()
        self.name = "SSIM2D"
    def forward(self, output, target):
        loss = (1 - self.ssim(output, target))
        return loss
    def getLossName(self):
        return self.name


class msssim2DLoss(nn.Module):
    """Calculate loss function of RGT"""
    def __init__(self):
        super(msssim2DLoss, self).__init__()
        self.ssim = ms_ssim_loss()
        self.name = "SSIM2D"
    def forward(self, output, target):
        loss = (1 - self.ssim(output, target))
        return loss
    def getLossName(self):
        return self.name


class mse3DLoss(nn.Module):
    """Calculate loss function of RGT"""
    def __init__(self):
        super(mse3DLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.name = "MSE"
    def forward(self, output, target):
        loss = self.mse(output, target)
        return loss
    def getLossName(self):
        return self.name
  


class diceloss1(torch.nn.Module):
    def __init__(self):
        super(diceloss1, self).__init__()
        self.name = 'diceloss1'
    def forward(self,pred, target):
       smooth = 1
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       dicevalue =  1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
       clamped_dice_loss = torch.clamp(dicevalue, min=0, max=1)
       return clamped_dice_loss
    def getLossName(self):
        return self.name



class diceloss(torch.nn.Module):
    def __init__(self):
        super(diceloss, self).__init__()
        self.name = 'diceloss'
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    def getLossName(self):
        return self.name



class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.name = 'BCEDiceLoss'
    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)

    def getLossName(self):
        return self.name



class SSIMmseloss(nn.Module):
    def __init__(self, a , b):
        super(SSIMmseloss, self).__init__()
        self.ssim = ssim2d()
        self.mse = mse3DLoss()
        self.name = "SSIMMSE"
        self.a = a
        self.b = b

    def forward(self, output, target):
        s = 1 - self.ssim(output, target)
        m = self.mse(output, target)
        loss = self.a*s+self.b*m
        return loss

    def getLossName(self):
        return self.name



class cross_loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = 'cross_loss'

    def forward(self, y_true, y_pred):
        smooth = 1e-5
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))

    def getLossName(self):
        return self.name



class dicecross(nn.Module):

    def __init__(self, a, b):
        super(dicecross, self).__init__()
        self.dice = diceloss()
        self.cross = cross_loss()
        self.name = 'ssimcross'
        self.a = a
        self.b = b

    def forward(self, y_true, y_pred):
        d = self.dice(y_true, y_pred)
        c = self.cross(y_true, y_pred)
        loss = self.a * d + self.b * c
        return loss

    def getLossName(self):
        return self.name



class dicessim(nn.Module):

    def __init__(self, a, b):
        super(dicessim, self).__init__()
        self.dice = diceloss()
        self.ssim = ssim2DLoss()
        self.name = 'dicessim'
        self.a = a
        self.b = b

    def forward(self, y_true, y_pred):
        d = self.dice(y_true, y_pred)
        c = self.ssim(y_true, y_pred)
        loss = self.a * d + self.b * c
        return loss

    def getLossName(self):
        return self.name



##### Adaptive tvMF Dice loss #####
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, kappa=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice
        return loss / self.n_classes