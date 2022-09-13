#!/usr/bin/env python3
import torchvision
import torch
from fire import Fire
from .extract import extract_features_

__all__ = ['extract_resnet18_imagenet_features']


def extract_resnet18_imagenet_features_(*args, **kwargs):
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)

    return extract_features_(*args, **kwargs, model=model, model_name='resnet18-imagenet')


if __name__ == '__main__':
    Fire(extract_resnet18_imagenet_features_)
