"""Helper classes to manage pytorch data."""
from dataclasses import dataclass, field
import itertools
from typing import Any, Callable, Optional, Sequence

from torch.utils.data import Dataset
import numpy as np
import torch


__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'


__all__ = ['ZipDataset', 'EncodedDataset']


class ZipDataset(Dataset):
    # TODO Upgrade typing to PEP 646 once Python 3.11 hits
    def __init__(
            self,
            *datasets: Dataset,
            strict: bool = True, flatten: bool = True
    ) -> None:
        """A dataset zipping multiple other datasets together.

        Args:
            datasets:  The datasets to zip together.
            strict:  Enforce the datasets to have the same length.  If
                false, then all datasets will be truncated to the shortest
                dataset's length.
            flatten:  Whether to combine the datasets into a single list.
        
        `flatten` can be used to control how the `ZipDataset`'s items will
        be combined:  Assume the `ZipDataset` consists of two subdatasets,
        each with scalar elements.  Then when using the `ZipDataset` with a
        Dataloader which loads the items in batches of size 64, then if
        `flatten` is true, the output will have the shape 64x2, while if
        `flatten` is false, it will have shape 64x2x1.
        """
        if strict:
            assert all(len(ds) == len(datasets[0]) for ds in datasets)
            self._len = len(datasets[0])
        else:
            self._len = min(len(ds) for ds in datasets)
        self._datasets = datasets
        self.flatten = flatten

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Any:
        if self.flatten:
            return tuple(itertools.chain.from_iterable(ds[index] for ds in self._datasets))
        else:
            return tuple(itertools.chain.from_iterable([ds[index]] for ds in self._datasets))

    def new_empty(self) -> 'ZipDataset':
        new_dss = [
            ds.new_empty() if hasattr(ds, 'new_empty') else ds
            for ds in self._datasets
        ]
        ds = ZipDataset(*new_dss, strict=False)
        return ds


@dataclass
class EncodedDataset(Dataset):
    """A dataset which first encodes its input data.
    
    This class is can be useful with classes such as fastai, where the
    encoder is saved as part of the model.
    """
    encode: Any
    """An sklearn encoding to encode the data with."""
    data: Sequence[Any] = field(default_factory=list)
    """Data to encode."""
    dtype: Optional[torch.dtype] = None
    """Type to cast the data into after encoding it."""

    def __getitem__(self, i: int) -> Any:
        encoded = torch.tensor(
            self.encode.transform(np.array(self.data[i]).reshape(-1, 1)),
            dtype=self.dtype)
        return encoded

    def __len__(self) -> int:
        return len(self.data)

    def new(self, data: Sequence[Any] = tuple()) -> 'EncodedDataset':
        """Create a dataset with the same encoding but different data."""
        return EncodedDataset(self.encode, data)

    def new_empty(self) -> 'EncodedDataset':
        """Create an empty dataset."""
        return self.new()