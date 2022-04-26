"""Attention-based deep multiple instance learning.

An implementation of

    arXiv:1802.04712
    Ilse, Maximilian, Jakub Tomczak, and Max Welling.
    "Attention-based deep multiple instance learning." 
    International conference on machine learning. PMLR, 2018.
"""

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
    SaveModelCallback, EarlyStoppingCallback, CSVLogger)
import pandas as pd

from marugoto.data import EncodedDataset, ZipDataset


__all__ = ['BagDataset', 'MILModel', 'Attention']


T = TypeVar('T')


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances."""
    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags.
    
    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x
    F, where N is the number of instances and F the number of features per
    instance.
    """
    bag_size: Optional[int] = None
    """The number of instances in each bag.
    
    For bags containing more instances, a random sample of `bag_size`
    instances will be drawn.  Smaller bags are padded with zeros.  If
    `bag_size` is None, all the samples will be used.
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # collect all the features
        feats = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, 'r') as f:
                feats.append(torch.from_numpy(f['feats'][:]))
        feats = torch.concat(feats)

        # sample a subset, if required
        if self.bag_size:
            return _to_fixed_size_bag(feats, bag_size=self.bag_size)
        else:
            return feats, len(feats)


def _to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> Tuple[torch.Tensor, int]:
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded = torch.cat((bag_samples,
                             torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1])))
    return zero_padded, min(bag_size, len(bag))


class MILModel(nn.Module):
    def __init__(
        self, n_feats: int, n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        """Create a new attention MIL model.

        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super().__init__()
        self.encoder = encoder or nn.Sequential(
            nn.Linear(n_feats, 256), nn.ReLU())
        self.attention = attention or Attention(256)
        self.head = head or nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, n_out))

    def forward(self, bags, lens):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        embeddings = self.encoder(bags)

        masked_attention_scores = self._masked_attention_scores(
            embeddings, lens)
        weighted_embedding_sums = (
            masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return torch.softmax(scores, dim=1)

    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.

        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size)
               .repeat(bs, 1)
               .to(attention_scores.device))

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask,
            attention_scores,
            torch.full_like(attention_scores, -1e10))
        return torch.softmax(masked_attention, dim=1)


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1))


def make_dataset(
    target_enc,
    bags: Sequence[Iterable[Path]], targets: Sequence[Any],
    bag_size: Optional[int] = None
) -> ZipDataset:
    """Creates a instance-wise dataset from MIL bag H5DFs."""
    assert len(bags) == len(targets), \
        'number of bags and ground truths does not match!'

    ds = ZipDataset(
        BagDataset(bags, bag_size=bag_size),
        EncodedDataset(target_enc, targets, dtype=torch.float32))

    return ds


def train(
    *,
    target_enc,
    train_bags: Sequence[Iterable[Path]],
    train_targets: Sequence[T],
    valid_bags: Sequence[Iterable[Path]],
    valid_targets: Sequence[T],
    n_epoch: int = 32,
    patience: int = 8,
    path: Optional[Path] = None,
) -> Learner:
    """Train a MLP on image features.

    Args:
        target_enc:  A scikit learn encoder mapping the targets to arrays
            (e.g. `OneHotEncoder`).
        train_bags:  H5s containing the bags to train on.
        train_targets:  The ground truths of the training bags.
        valid_bags:  H5s containing the bags to validate on.
        train_targets:  The ground thruths of the validation bags.
    """
    train_ds = make_dataset(target_enc, train_bags, train_targets, bag_size=512)
    valid_ds = make_dataset(target_enc, valid_bags, valid_targets, bag_size=None)

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    batch = train_dl.one_batch()

    model = MILModel(batch[0].shape[-1], batch[-1].shape[-1])

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
    assert test_df.PATIENT.nunique() == len(test_df), \
        'duplicate patients!'

    target_enc = learn.dls.train.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]

    test_ds = make_dataset(
        target_enc=target_enc,
        bags=test_df.slide_path.values,
        targets=test_df[target_label].values,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count())

    patient_preds, patient_targs = learn.get_preds(dl=test_dl)

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