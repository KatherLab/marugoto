import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# https://github.com/jiawei-ren/BalancedMSE
class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    target = target[0]
    #print("Calculating BMC loss...")
    #breakpoint()
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var) #.detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss




class WeightedMSELoss(nn.Module):
    def __init__(self, **kwargs): #weights=None,
        super().__init__()
        #self.weights = weights

    def forward(self, inputs, targets): #added weights as targets[1]
        # weights = self.weights
        #breakpoint()
        weights = targets[1] + 1
        loss = (inputs - targets[0]) ** 2
        if weights is not None:
            loss = loss*weights #remove squeeze()
        loss = torch.mean(loss)
        return loss

class WeightedL1Loss(nn.Module):
    def __init__(self, **kwargs): #weights=None,
        super().__init__()
        #self.weights = weights

    def forward(self, inputs, targets): #added weights as targets[1]
        # weights = self.weights
        #breakpoint()
        weights = targets[1] + 1
        loss = F.l1_loss(inputs, targets[0], reduction='none')
        if weights is not None:
            loss = loss*weights #remove squeeze()
        loss = torch.mean(loss)
        return loss

class WeightedHuberLoss(nn.Module):
    def __init__(self, **kwargs): #weights=None,
        super().__init__()
        #self.weights = weights

    def forward(self, inputs, targets): #added weights as targets[1]
        # weights = self.weights
        #breakpoint()
        weights = targets[1] + 1
        loss = F.huber_loss(inputs, targets[0], reduction='none', delta=1.0)
        if weights is not None:
            loss = loss*weights #remove squeeze()
        loss = torch.mean(loss)
        return loss

class WeightedFocalMSELoss(nn.Module):
    def __init__(self, activate='sigmoid', beta=.2, gamma=1, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.activate = activate

    def forward(self, inputs, targets):
        gamma = self.gamma
        beta = self.beta
        activate = self.activate
        weights = targets[1] + 1
        targets = targets[0]

        loss = (inputs - targets) ** 2
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
            (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
        if weights is not None:
            loss = loss.squeeze()*weights
        loss = torch.mean(loss)
        return loss


class WeightedFocalL1Loss(nn.Module):
    def __init__(self, weights=None, activate='sigmoid', beta=.2, gamma=1, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.activate = activate

    def forward(self, inputs, targets):
        gamma = self.gamma
        beta = self.beta
        activate = self.activate
        weights = targets[1] + 1
        targets = targets[0]

        #breakpoint()
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
            (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
        if weights is not None:
            loss = loss.squeeze()*weights
        loss = torch.mean(loss)
        return loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
