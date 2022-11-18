import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

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


def mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True
):
    """Mean squared error regression loss.

    Read more in the :ref:`User Guide <mean_squared_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.612...
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)
    0.708...
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.822...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825...
    """
    #ONLY LOOK AT THE TARGET, NOT THE WEIGHT
    y_pred = y_pred[0]


    output_errors = np.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)

    if not squared:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


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
