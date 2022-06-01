# %%
import os
import re
from typing import Optional
import torch
from torch.utils.data import Dataset
from pathlib import Path
import PIL
from torchvision import models, transforms
from torch import nn
import numpy as np
from tqdm import tqdm
import h5py

__all__ = ['extract_features_']


# %%
class SlideTileDataset(Dataset):
    def __init__(self, slide_dir: Path, transform=None, *, repetitions: int = 1) -> None:
        assert repetitions >= 1, 'at least one repetition of the dataset required'
        self.tiles = list(slide_dir.glob('*.jpg'))*repetitions
        assert self.tiles, f'no tiles found in {slide_dir}'
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.open(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image


def _get_coords(filename) -> Optional[np.ndarray]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None


def extract_features_(
    *slide_tile_paths: Path, outdir: Path, augmented_repetitions: int = 0
) -> None:
    """Extracts features from slide tiles.
    
    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model = model.half().eval().to(device)
    normal_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    augmenting_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomVerticalFlip(p=.5),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=.1, contrast=.2, saturation=.25, hue=.125)], p=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for slide_tile_path in tqdm(slide_tile_paths):
        slide_tile_path = Path(slide_tile_path)
        if (h5outpath := outdir/f'{slide_tile_path.name}.h5').exists():
            print(f'{h5outpath} already exists.  Skipping...')
            continue
        if not next(slide_tile_path.glob('*.jpg'), False):
            print(f'No tiles in {slide_tile_path}.  Skipping...')
            continue

        ds = SlideTileDataset(slide_tile_path, normal_transform)
        _extract(ds, model, device, h5outpath)

        if augmented_repetitions:
            ds = SlideTileDataset(
                slide_tile_path, augmenting_transform, repetitions=augmented_repetitions-1)
            _extract(ds, model, device, h5outpath)


def _extract(ds, model, device, h5outpath):
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=os.cpu_count(), drop_last=False)
    feats = []
    for batch in tqdm(dl, leave=False):
        feats.append(model(batch.half().to(device)).detach())

    with h5py.File(h5outpath, 'w') as f:
        f['coords'] = [_get_coords(fn) for fn in ds.tiles]
        f['feats'] = torch.concat(feats).cpu().numpy()


if __name__ == '__main__':
    import fire
    fire.Fire(extract_features_)