#!/usr/bin/env python3
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Resnet18 imagenet features from slide."
    )
    parser.add_argument(
        "slide_tile_paths",
        metavar="SLIDE_TILE_DIR",
        type=Path,
        nargs="+",
        help="A directory with tiles from a slide.",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, required=True, help="Path to save the features to."
    )
    parser.add_argument(
        "--augmented-repetitions",
        type=int,
        default=0,
        help="Also save augmented feature vectors.",
    )
    args = parser.parse_args()
    print(f"{args=}")

import torchvision
import torch
from .extract import extract_features_

__all__ = ["extract_resnet18_imagenet_features"]


def extract_resnet18_imagenet_features_(slide_tile_paths, **kwargs):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)

    return extract_features_(
        slide_tile_paths=slide_tile_paths,
        model=model,
        model_name="resnet18-imagenet",
        **kwargs,
    )


if __name__ == "__main__":
    extract_resnet18_imagenet_features_(**vars(args))
