from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os

from fastai.vision.all import (
    Learner,
    DataLoader,
    DataLoaders,
    RocAuc,
    SaveModelCallback,
    CSVLogger,
)
import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import numpy.typing as npt
import pandas as pd

from marugoto.data import SKLearnEncoder

from .data import make_dataset
from .model import MILModel


__all__ = ["train", "deploy"]


T = TypeVar("T")


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, npt.NDArray],
    add_features: Iterable[Tuple[SKLearnEncoder, npt.NDArray]] = [],
    valid_idxs: npt.NDArray[np.int_],
    n_epoch: int = 32,
    path: Optional[Path] = None,
    drop_last: Optional[bool] = False,
    batch_size: Optional[int] = 64,
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    target_enc, targs = targets

    train_ds = make_dataset(
        bags=bags[~valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[(enc, vals[~valid_idxs]) for enc, vals in add_features],
        bag_size=512,
    )

    valid_ds = make_dataset(
        bags=bags[valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[valid_idxs]),
        add_features=[(enc, vals[valid_idxs]) for enc, vals in add_features],
        bag_size=None,
    )

    # build dataloaders
    if drop_last:
        assert len(train_ds)<=batch_size, f"Error: batch size ({batch_size}) is higher than data set length ({len(train_ds)})!!" 
        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )
    batch = train_dl.one_batch()

    model = MILModel(batch[0].shape[-1], batch[-1].shape[-1])

    # weigh inversely to class occurances
    counts = pd.value_counts(targs[~valid_idxs])
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32
    )
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl)
    learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=path)

    cbs = [
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs)

    return learn


def deploy(
    test_df: pd.DataFrame,
    learn: Learner,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None,
    cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), "duplicate patients!"
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
        bag_size=None,
    )

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )

    # removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict(
        {
            "PATIENT": test_df.PATIENT.values,
            target_label: test_df[target_label].values,
            **{
                f"{target_label}_{cat}": patient_preds[:, i]
                for i, cat in enumerate(categories)
            },
        }
    )

    # calculate loss
    patient_preds = patient_preds_df[
        [f"{target_label}_{cat}" for cat in categories]
    ].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1)
    )
    patient_preds_df["loss"] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs), reduction="none"
    )

    patient_preds_df["pred"] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[
        [
            "PATIENT",
            target_label,
            "pred",
            *(f"{target_label}_{cat}" for cat in categories),
            "loss",
        ]
    ]
    patient_preds_df = patient_preds_df.sort_values(by="loss")

    return patient_preds_df
