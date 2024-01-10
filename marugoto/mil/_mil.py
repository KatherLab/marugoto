from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os

import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, EarlyStoppingCallback)
import pandas as pd
import numpy as np

from marugoto.data import SKLearnEncoder

from .data import make_dataset
from .transformer import Transformer
from .ViT import ViT
from .loss import mean_squared_error


__all__ = ['train', 'deploy']


T = TypeVar('T')


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: np.ndarray,
    add_features: Iterable[Tuple[SKLearnEncoder, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    n_epoch: int = 32,
    patience: int = 16,
    path: Optional[Path] = None,
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    train_ds = make_dataset(
        bags=bags[~valid_idxs],
        targets=targets[~valid_idxs],
        add_features=[
            (enc, vals[~valid_idxs])
            for enc, vals in add_features],
        bag_size=4096) # set to None, usually 512-ish for batch_size > 1

    valid_ds = make_dataset(
        bags=bags[valid_idxs],
        targets=targets[valid_idxs],
        add_features=[
            (enc, vals[valid_idxs])
            for enc, vals in add_features],
        bag_size=None)

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=32) # set to 1 for regression
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=32)
    batch = train_dl.one_batch()

    # 1 as output dim for regression, 768 for ctranspath dims
    model = ViT(num_classes=1, input_dim=768)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) #

    loss_func = nn.MSELoss()

    dls = DataLoaders(train_dl, valid_dl) #
    learn = Learner(dls, model, loss_func=loss_func,lr=.0001, wd=0.01,
                    metrics=[mean_squared_error], path=path)

    cbs = [
        SaveModelCallback(fname=f'best_valid'),
        EarlyStoppingCallback(monitor='valid_loss',
                              min_delta=0.0000001, patience=8),
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
        targets=(test_df[target_label].values).reshape(-1,1),
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=8) #shuffle=True #drop_last=True


    patient_preds, patient_targs = learn.get_preds(dl=test_dl)

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
