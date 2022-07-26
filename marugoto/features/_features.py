#!/usr/bin/env python3
"""Train a network on MIL h5 bag features."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence, Optional, TypeVar, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from fastai.vision.all import (
    create_head, Learner, RocAuc,
    SaveModelCallback, EarlyStoppingCallback, CSVLogger,
    DataLoader, DataLoaders)

from marugoto.data import ZipDataset, EncodedDataset


__all__ = [
    'make_dataset', 'H5TileDataset', 'train', 'deploy']


T = TypeVar('T')
PathLike = Union[str, Path]


def make_dataset(
    target_enc,
    bags: Sequence[Path], targets: Sequence[Any],
    seed: Optional[int] = 0
) -> ZipDataset:
    """Creates a instance-wise dataset from MIL bag H5DFs."""
    assert len(bags) == len(targets), \
        'number of bags and ground truths does not match!'
    tile_ds: ConcatDataset = ConcatDataset(
        H5TileDataset(h5, seed=seed) for h5 in bags)
    lens = np.array([len(ds) for ds in tile_ds.datasets])
    ys = np.repeat(targets, lens)
    ds = ZipDataset(
        tile_ds,
        EncodedDataset(target_enc, ys, dtype=torch.float32)) #     type: ignore
    return ds


@dataclass
class H5TileDataset(Dataset):
    """A dataset containing the instances of a MIL bag."""
    h5path: Path
    """H5DF file to take the bag tile features from.

    The file has to contain a dataset 'feats' of dimension NxF, where N is
    the number of tiles and F the dimension of the tile's feature vector.
    """
    tile_no: Optional[int] = 256
    """Number of tiles to sample (with replacement) from the bag.

    If `tile_no` is `None`, _all_ the bag's tiles will be taken.
    """
    seed: Optional[int] = None
    """Seed to initialize the RNG for sampling.

    If `tile_no` is `None`, this option has no effect and all the bag's
    tiles will be given in the same order as in the h5.
    """

    def __post_init__(self):
        assert not self.seed or self.tile_no, \
            '`seed` must not be set if `tile_no` is `None`.'

    def __getitem__(self, index) -> torch.Tensor:
        with h5py.File(self.h5path, mode='r') as f:
            if self.tile_no:
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                index = torch.randint(
                    len(f['feats']), (self.tile_no or len(f['feats']),))[index]
            return torch.tensor(f['feats'][index]).unsqueeze(0)

    def __len__(self):
        if self.tile_no:
            return self.tile_no

        with h5py.File(self.h5path, mode='r') as f:
            len(f['feats'])



def train(
    *,
    target_enc,
    train_bags: Sequence[Path],
    train_targets: Sequence[T],
    valid_bags: Sequence[Path],
    valid_targets: Sequence[T],
    n_epoch: int = 32,
    patience: int = 4,
    path: Optional[Path] = None,
) -> Learner:
    """Train a MLP on image features.

    Args:
        target_enc:  A scikit learn encoder mapping the targets to arrays
            (e.g. `OneHotEncoder`).
        train_bags:  H5s containing the bags to train on (cf.
            `marugoto.mil`).
        train_targets:  The ground truths of the training bags.
        valid_bags:  H5s containing the bags to validate on.
        train_targets:  The ground thruths of the validation bags.
    """
    print(type(target_enc))
    train_ds = make_dataset(target_enc, train_bags, train_targets)
    valid_ds = make_dataset(target_enc, valid_bags, valid_targets, seed=0)

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    valid_dl = DataLoader(
        valid_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count())
    batch = train_dl.one_batch()

    model = create_head(batch[0].shape[-1], batch[1].shape[-1], concat_pool=False)[1:]

    # weigh inversely to class occurances
    counts = pd.value_counts(train_targets)
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl)
    learn = Learner(dls, model, loss_func=loss_func,
                    metrics=[RocAuc()], path=path)

    cbs = [
        SaveModelCallback(monitor='roc_auc_score', fname=f'best_valid'),
        EarlyStoppingCallback(monitor='roc_auc_score',
                              min_delta=0.01, patience=patience),
        CSVLogger()]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-3, cbs=cbs)

    return learn


def deploy(test_df, learn, target_label):
    target_enc = learn.dls.train.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]

    test_ds = make_dataset(
        target_enc=target_enc,
        bags=test_df.slide_path.values,
        targets=test_df[target_label].values)

    test_dl = DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count())

    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax())

    # create tile wise result dataframe
    tiles_per_slide = [len(ds) for ds in test_ds._datasets[0].datasets]
    tile_score_df = pd.DataFrame.from_dict({
        'PATIENT': np.repeat(test_df.PATIENT.values, tiles_per_slide),
        **{f'{target_label}_{cat}': patient_preds[:, i]
           for i, cat in enumerate(categories)}})

    # calculate mean patient score, merge with ground truth label
    patient_preds_df = tile_score_df.groupby('PATIENT').mean().reset_index()
    patient_preds_df = patient_preds_df.merge(
        test_df[['PATIENT', target_label]].drop_duplicates(), on='PATIENT')

    # calculate loss
    patient_preds = patient_preds_df[[
        f'{target_label}_{cat}' for cat in categories]].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs),
        reduction='none')

    patient_preds_df['pred'] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        *(f'{target_label}_{cat}' for cat in categories),
        'loss']]
    patient_preds_df = patient_preds_df.sort_values(by='loss')
    return patient_preds_df
