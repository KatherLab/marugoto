#!/usr/bin/env python3
"""Train a network on MIL h5 bag features."""
from dataclasses import dataclass
import json
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
    DataLoader, DataLoaders, load_learner)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .data import ZipDataset, EncodedDataset


__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'

__changelog__ = {
    '0.2.0': 'Add CLI for training and deploying models.',
}

__all__ = [
    'make_dataset', 'H5TileDataset', 'train',
    'train_categorical_model_cli', 'deploy_categorical_model_cli']


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
        EncodedDataset(target_enc, ys, dtype=torch.float32))    # type: ignore
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


def train_categorical_model_cli(
        clini_excel: PathLike,
        slide_csv: PathLike,
        feature_dir: PathLike,
        target_label: str,
        output_path: PathLike,
        categories=None,
) -> None:
    """Train a categorical model on a cohort's tile's features.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
        feature_dir:  Path containing the features.
        output_path:  File to save model in.
    """
    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    from datetime import datetime
    info = {
        'description': 'training on tile features',
        'clini': str(clini_excel.absolute()),
        'slide': str(slide_csv.absolute()),
        'feature_dir': str(feature_dir.absolute()),
        'target_label': str(target_label),
        'output_path': str(output_path.absolute()),
        'datetime': datetime.now().astimezone().isoformat()}

    model_path = output_path/'export.pkl'
    if model_path.exists():
        print(f'{model_path} already exists. Skipping...')
        return

    clini_df = pd.read_excel(clini_excel)
    slide_df = pd.read_csv(slide_csv)
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    if not categories:
        categories = df[target_label].unique()
        print(f'Inferred {categories=}')
    categories = np.array(categories)
    info['categories'] = list(categories)

    df = df[df[target_label].isin(categories)]

    slides = set(feature_dir.glob('*.h5'))
    # remove slides we don't have
    slide_df = pd.DataFrame(slides, columns=['slide_path'])
    slide_df['FILENAME'] = slide_df.slide_path.map(lambda p: p.stem)
    df = df.merge(slide_df, on='FILENAME')

    print('Overall distribution')
    print(df[target_label].value_counts())
    info['class distribution'] = {'overall': {
        k: int(v) for k, v in df[target_label].value_counts().items()}}

    # Split off validation set
    patient_df = df.groupby('PATIENT').first()

    train_patients, valid_patients = train_test_split(
        patient_df.index, stratify=patient_df[target_label])
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]

    info['class distribution']['training'] = {
        k: int(v) for k, v in train_df[target_label].value_counts().items()}
    info['class distribution']['validation'] = {
        k: int(v) for k, v in valid_df[target_label].value_counts().items()}

    target_enc = OneHotEncoder(sparse=False).fit(categories.reshape(-1, 1))

    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)

    learn = train(
        target_enc=target_enc,
        train_bags=train_df.slide_path.values,
        train_targets=train_df[target_label].values,
        valid_bags=valid_df.slide_path.values,
        valid_targets=valid_df[target_label].values,
        path=output_path)

    learn.export(model_path)


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
        valid_bags:  H5s containing the bags to train on.
        train_targets:  The ground thruths of the validation bags.
    """
    train_ds = make_dataset(target_enc, train_bags, train_targets)
    valid_ds = make_dataset(target_enc, valid_bags, valid_targets, seed=0)

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    valid_dl = DataLoader(
        valid_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count())

    model = nn.Sequential(
        create_head(512, 2)[1:],
        nn.Softmax(dim=1))

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


def deploy_categorical_model_cli(
        clini_excel: PathLike,
        slide_csv: PathLike,
        feature_dir: PathLike,
        target_label: str,
        model_path: PathLike,
        output_path: PathLike,
) -> None:
    """Deploy a categorical model on a cohort's tile's features.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        feature_dir:  Path containing the features.
        model_path:  Path of the model to deploy.
        output_path:  File to save model in.
    """
    feature_dir = Path(feature_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)
    if (preds_csv := output_path/'patient-preds.csv').exists():
        print(f'{preds_csv} already exists!  Skipping...')
        return

    clini_df = pd.read_excel(clini_excel)
    slide_df = pd.read_csv(slide_csv)
    test_df = clini_df.merge(slide_df, on='PATIENT')

    learn = load_learner(model_path)
    target_enc = learn.dls.train.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]

    # remove uninteresting
    test_df = test_df[test_df[target_label].isin(categories)]
    # remove slides we don't have
    slides = set(feature_dir.glob('*.h5'))
    slide_df = pd.DataFrame(slides, columns=['slide_path'])
    slide_df['FILENAME'] = slide_df.slide_path.map(lambda p: p.stem)
    test_df = test_df.merge(slide_df, on='FILENAME')

    test_ds = make_dataset(
        target_enc=target_enc,
        bags=test_df.slide_path.values,
        targets=test_df.isMSIH.values)

    test_dl = DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count())

    patient_preds, patient_targs = learn.get_preds(dl=test_dl)

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
    output_path.mkdir(parents=True, exist_ok=True)
    patient_preds_df.to_csv(preds_csv, index=False)


if __name__ == '__main__':
    from fire import Fire
    Fire({
        'train': train_categorical_model_cli,
        'deploy': deploy_categorical_model_cli,
    })
