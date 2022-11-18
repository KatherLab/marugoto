from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os

import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, GradientAccumulation)
#from fastai.callback import tensorboard

#from fastai.callback.tensorboard import TensorBoardCallback
import pandas as pd
import numpy as np


#Deep imbalanced regression
from .loss import WeightedMSELoss, WeightedL1Loss, WeightedHuberLoss
from fastai.optimizer import OptimWrapper
from fastai.optimizer import SGD

from collections import Counter
from scipy.ndimage import convolve1d
from .utils import get_lds_kernel_window
#######################################


from marugoto.data import FunctionTransformer
from sklearn.metrics import r2_score
from .loss import mean_squared_error
#from sklearn.metrics.pairwise import manhattan_distances

# from sklearn.neural_network import MLPRegressor #just to test a simpler model
# from fastai.vision.all import * #same as avive

from .data import make_dataset
from .model import MILModel


__all__ = ['train', 'deploy']


T = TypeVar('T')

#CHANGED
def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: np.ndarray,
    add_features: Iterable[Tuple[FunctionTransformer, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    n_epoch: int = 25, #32
    patience: int = 8,
    path: Optional[Path] = None,
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    targs = targets

    def get_bin_idx(x, bins):
    #TODO: find optimal binning strategy
        '''
        x is a continuous variable (normalised) between 0-1
        to get the bins, x is rounded to its nearest decimal,
        and then multiplied by ten. Totalling in 11 bins which
        will be weighed accordingly to its frequency within
        '''
        label = None
        for i, bin in enumerate(bins):
            if x <= bin:
                label = i
                break


        return label


    def weighting_continuous_values(labels) -> torch.FloatTensor:
        

        edges = np.histogram_bin_edges(labels, bins='auto')
        bin_index_per_label = np.array([get_bin_idx(label, edges) for label in labels])
        # calculate empirical (original) label distribution: [Nb,]
        # "Nb" is the number of bins
        Nb = max(bin_index_per_label) + 1

        #i.e., np.histogram(bin_index_per_label)
        unique, counts = np.unique(bin_index_per_label, return_counts=True)
        num_samples_of_bins = dict(zip(unique, counts))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
        # calculate effective label distribution: [Nb,]
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = torch.FloatTensor([np.float32(1 / x) for x in eff_num_per_label])

        return weights


    weights = weighting_continuous_values(targs).reshape(-1,1)



    train_ds = make_dataset(
        bags=bags[~valid_idxs.values],
        targets= (targs[~valid_idxs.values], weights[~valid_idxs.values]),
        add_features=[
            (enc, vals[~valid_idxs.values])
            for enc, vals in add_features],
        bag_size=None) #512


    #CHANGED
    valid_ds = make_dataset(
        bags=bags[valid_idxs.values],
        targets=(targs[valid_idxs.values], weights[valid_idxs.values]),
        add_features=[
            (enc, vals[valid_idxs.values])
            for enc, vals in add_features],
        bag_size=None) #None


    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=1) #batch_size=64, shuffle=True drop_last=True
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=1) #batch_size=1, shuffle=False , drop_last=True

    #Graziani et al: batch_size_bag = 1, shuffle=True for both
    batch = train_dl.one_batch()


    #added extra [0] because of the new tuple structure
    model = MILModel(batch[0][0].shape[-1], 1) #batch[-1].shape[-1]

    # print(model)
    # MILModel(
    # (encoder): Sequential(
    #     (0): Linear(in_features=2048, out_features=256, bias=True)
    #     (1): ReLU()
    # )
    # (attention): Sequential(
    #     (0): Linear(in_features=256, out_features=128, bias=True)
    #     (1): Tanh()
    #     (2): Linear(in_features=128, out_features=1, bias=True)
    # )
    # (head): Sequential(
    #     (0): Flatten(start_dim=1, end_dim=-1)
    #     (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (2): Dropout(p=0.5, inplace=False)
    #     (3): Linear(in_features=256, out_features=1, bias=True)
    # )
    # )
    
    #for imbalanced regression
    loss_func = WeightedMSELoss()

    dls = DataLoaders(train_dl, valid_dl)
    
    #SGD instead of Adam standard, from Graziani et al.
    #def opt_func(params, **kwargs): return OptimWrapper(SGD(params, lr=.0001, mom=.9, wd=0.01))

    #mean squared error metric is 'handmade' from .loss file
    learn = Learner(dls, model, loss_func=loss_func, lr=.0001, wd=0.01,
                    metrics=[mean_squared_error], path=path)


    cbs = [
        SaveModelCallback(fname=f'best_valid'),
        #EarlyStoppingCallback(monitor='roc_auc_score',
        #                      min_delta=0.01, patience=patience),
        #GradientAccumulation(n_acc=64),
        #TensorBoardCallback(),
        CSVLogger()]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs)

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'


    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    #CHANGED
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    #CHANGED
    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        #weights=weights.reshape(-1,1),
        targets=(((test_df[target_label].values)).reshape(-1,1), np.ones((test_df[target_label].values).shape).reshape(-1,1)), #(target_enc, )
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0) #shuffle=True #drop_last=True


    patient_preds, patient_targs = learn.get_preds(dl=test_dl)
    patient_targs = patient_targs[0]

    # make into DF w/ ground truth
    #CHANGED
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values})

    patient_preds_df['loss'] = F.mse_loss(
        patient_preds.clone().detach(), patient_targs.clone().detach(),
        reduction='none')

    patient_preds_df['pred'] = patient_preds

    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        'loss']]

    return patient_preds_df
