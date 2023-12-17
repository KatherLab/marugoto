from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os
import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger)
import pandas as pd

from data import SKLearnEncoder
from matplotlib import pyplot as plt
from .data import make_dataset
from .model import MILModel
from loss import concordance_index, cox_loss

__all__ = ['train', 'deploy']


T = TypeVar('T')


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, Sequence[Any]],
    add_features: Iterable[Tuple[SKLearnEncoder, Sequence[Any]]] = [],
    valid_idxs: Iterable[bool],
    n_epoch: int = 32,
    batch_size: int = 16,
    num_workers: int = min(os.cpu_count(), 8),
    lr_max: float = 0.0001,
    lr_find_dir: Path = Path.cwd(),
    path: Optional[Path] = None,
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    # target_enc, targs = targets
    targs = targets
    train_ds = make_dataset(
        bags=bags[~valid_idxs],
        # targets=(target_enc, targs[~valid_idxs]),
        targets=targs[~valid_idxs],
        add_features=[
            (enc, vals[~valid_idxs])
            for enc, vals in add_features],
        bag_size=512)

    valid_ds = make_dataset(
        bags=bags[valid_idxs],
        # targets=(target_enc, targs[valid_idxs]),
        targets=targs[valid_idxs],
        add_features=[
            (enc, vals[valid_idxs])
            for enc, vals in add_features],
        bag_size=None)

    # build dataloaders
    drop_last = True # train_ds._len % batch_size <= batch_size//2#
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last,
        num_workers=num_workers)
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False,
        num_workers=os.cpu_count())
    batch = train_dl.one_batch()

    model = MILModel(batch[0].shape[-1], 1)

    # weigh inversely to class occurances
    # counts = pd.value_counts(targs[~valid_idxs])
    # weight = counts.sum() / counts
    # weight /= weight.sum()
    # # reorder according to vocab
    # weight = torch.tensor(
    #     list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32)

    # loss_func = nn.CrossEntropyLoss(weight=weight)
    loss_func = cox_loss
    dls = DataLoaders(train_dl, valid_dl)
    learn = Learner(dls, model, loss_func=loss_func,
                    metrics=[concordance_index], path=path)

    cbs = [
        SaveModelCallback(fname=f'best_valid'),
        CSVLogger()]

    if not lr_max:
        print('Searching learning rate...')
        suggested_lrs = learn.lr_find()
        lr_max = suggested_lrs.valley
        plt.savefig(lr_find_dir/'lr_find.jpg')
        plt.close('all')
        print(f'\n Using lr_max of {lr_max:.4}!\n')
    else:
        print(f'Using hard coded learing rate of {lr_max}.')

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=lr_max, cbs=cbs)

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'
    # assert (len(add_label)
    #        == (n := len(learn.dls.train.dataset._datasets[-2]._datasets))), \
    #    f'not enough additional feature labels: expected {n}, got {len(add_label)}'
    if target_label is None:
        target_label = learn.target_label
    if cat_labels is None:
        cat_labels = learn.cat_labels
    if cont_labels is None:
        cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count())

    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}': patient_preds[:, i]
            for i, cat in enumerate(categories)}})

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
