"""Attention-based deep multiple instance learning.

An implementation of

    arXiv:1802.04712
    Ilse, Maximilian, Jakub Tomczak, and Max Welling.
    "Attention-based deep multiple instance learning." 
    International conference on machine learning. PMLR, 2018.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from pathlib import Path

import h5py
import torch
from torch import nn
from torch.utils.data import Dataset


__author__ = "Marko van Treeck"
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'


__all__ = ['BagDataset', 'MILModel', 'Attention']


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances."""
    bags: Sequence[Path]
    """The `.h5` files containing the bags.
    
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
        with h5py.File(self.bags[index], 'r') as f:
            feats = torch.from_numpy(f['feats'][:])
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
