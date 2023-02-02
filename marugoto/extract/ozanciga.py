#!/usr/bin/env python3
import hashlib
import torchvision
import torch
from fire import Fire
from .extract import extract_features_

# use like this:
# python -m marugoto.extract.ozanciga \
#   --checkpoint-path ~jxiaofeng/Downloads/tenpercent_resnet18.ckpt \
#   --outdir /run/media/jxiaofeng/Sirius_03_empty/TCGA_features/TCGA-CRC-DX-features/ozanciga-augmented \
#   --augmented-repetitions 3
#   /run/media/jxiaofeng/Sirius_03_empty/TCGA_BLOCKS/TCGA-CRC-DX-BLOCKS/*


__all__ = ["extract_ozanciga_features"]


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print("No weight could be loaded..")
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


def extract_ozanciga_features_(checkpoint_path: str, *args, **kwargs):
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
        == "61dc3ce7273f4ecb2960c3df31b67e36c38e4f30d9622b42496f842752992117"
    )

    model = torchvision.models.__dict__["resnet18"](pretrained=False)

    state = torch.load(checkpoint_path, map_location="cuda:0")

    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(
            key
        )

    model = load_model_weights(model, state_dict)
    model.fc = torch.nn.Identity()

    return extract_features_(
        *args, **kwargs, model=model.cuda(), model_name="ozanciga-61dc3ce7"
    )


if __name__ == "__main__":
    Fire(extract_ozanciga_features_)
