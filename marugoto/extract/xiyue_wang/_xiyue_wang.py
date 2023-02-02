#!/usr/bin/env python3
"""Extract features with pretrained model from RetCCL: Clustering-guided Contrastive Learning for Whole-slide Image Retrieval

https://github.com/Xiyue-Wang/RetCCL

use like this:
    python -m marugoto.extract.xiyue_wang \
        --checkpoint-path ~/Downloads/best_ckpt.pth \
        --outdir ~/TCGA_features/TCGA-CRC-DX-features/xiyue-wang \
        /mnt/TCGA_BLOCKS/TCGA-CRC-DX-BLOCKS/*
"""
# %%
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from marugoto.extract.extract import extract_features_
from .RetCLL import ResNet

# %%


def extract_xiyuewang_features_(
    *slide_tile_paths: Path, checkpoint_path: str, **kwargs
):
    """Extracts features from slide tiles.

    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL>.
    """
    # calculate checksum of model
    sha256 = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert (
        sha256.hexdigest()
        == "931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff"
    )

    model = ResNet.resnet50(
        num_classes=128, mlp=False, two_branch=False, normlinear=True
    )
    pretext_model = torch.load(checkpoint_path)
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)

    return extract_features_(
        slide_tile_paths=slide_tile_paths,
        model=model.cuda(),
        model_name="xiyuewang-retcll-931956f3",
        **kwargs
    )
