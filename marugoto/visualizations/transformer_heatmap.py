#!/usr/bin/env python3

__author__ = "Marko van Treeck"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__maintainer__ = ["Marko van Treeck", "Omar El Nahhas"]
__email__ = "markovantreeck@gmail.com"

# %%
from collections import namedtuple
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fastai.vision.all import load_learner
from matplotlib import pyplot as plt
from PIL import Image
# %%
learn = load_learner(
    "export.pkl"
)
feature_name = "1549498.h5"
f_hname = '1549498'
output_folder = "/heatmaps"

learn.model.cuda().eval()

true_class_index = 0  # TODO FIND OUT WHICH ONE IS THE TRUE CLASS QQ

# load slide
# TODO loop over h5s
f = h5py.File(feature_name)
coords = f["coords"][:]

xs = np.sort(np.unique(coords[:, 0]))
stride = np.min(xs[1:] - xs[:-1])


# forward features through transformer
feats = torch.tensor(f["feats"][:]).cuda().float()
feats.requires_grad = True
scores = learn.model(feats.unsqueeze(0)).squeeze()
# breakpoint()

#  Visualize the grad-cam attention only
scores.backward()
gradcam = (feats.grad * feats).abs().sum(-1)


def vals_to_im(scores, coords, stride):
    # breakpoint()
    size = coords.max(0)[::-1] #// stride + 1
    if scores.ndimension() == 1:
        im = np.zeros(size)
    elif scores.ndimension() == 2:
        im = np.zeros((*size, scores.size(-1)))
    else:
        raise ValueError(f"{scores.ndimension()=}")
    for score, c in zip(scores, coords[:]): #, strict=True
        x, y = c[0], c[1]
        im[y:(y+stride), x:(x+stride)] = score.cpu().detach().numpy()
    return im


att_im = vals_to_im(gradcam, coords, stride)
fig, ax = plt.subplots()
ax.imshow(att_im, alpha=np.float32(att_im != 0))
ax.axis("off")
# TODO save image

# Visualize tile-wise scores, with attention as alpha
# Each tile is treated as if it were the only tile in the slide
# to get a localized prediction
# This leads to more extreme predictions
tile_logits = learn.model(feats.unsqueeze(-2))
tile_scores = F.softmax(tile_logits, 0)
score_im = vals_to_im(tile_scores.detach().cpu(), coords, stride)
fig, ax = plt.subplots()
ax.imshow(
    score_im[:, :, 0],
    vmin=0,
    vmax=1,
    cmap="bwr",
    alpha=att_im / att_im.max(),
)
ax.axis("off")
# TODO save image

# We'll register a forward hook to extract the
# query, key and value of each token in the forward pass
# of the transformer
q, k = None, None


def save_qkv(_module, _args, output):
    global q, k
    qkv = output.chunk(3, dim=-1)
    n_heads = 8
    # this next line is stolen from Sofia's code,
    # I don't quite get it...
    q, k, _ = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=n_heads), qkv)

for transformer_layer in range(2):
    img_coll=[]
    # fig, axs = plt.subplots(2, 4, figsize=(4, 2), dpi=300)
    # for attention_head_i, ax in enumerate(axs.reshape(-1)):
    for attention_head_i in range(8): #8heads in model
        # for some reason we have to reload the features every iteration
        # or we get different results... Investigate!
        feats = torch.tensor(f["feats"][:]).cuda().float()
        feats.requires_grad = True
        embedded = learn.fc(feats.unsqueeze(0).float())
        with_class_token = torch.cat([learn.cls_token, embedded], dim=1)

        with learn.model.transformer.layers[transformer_layer][
            0
        ].fn.to_qkv.register_forward_hook(save_qkv):
            transformed = learn.transformer(with_class_token)[:, 0]
        a = F.softmax(q @ k.transpose(-2, -1) * 0.125, dim=-1)

        # calculate attention gradcam
        a[0, attention_head_i, 0, 1:].sum().backward()
        #TODO think long and hard about this `abs`
        gradcam = (feats.grad * feats).abs().sum(-1)
        X = vals_to_im(gradcam, coords, stride)

        if (X.max() - X.min()) > 0:
            X_std = (X - X.min()) / (X.max() - X.min())
        else:
            continue
        X_scaled = X_std * (255)
        img_coll.append(X_scaled) #for the aggregated heatmap later, like Firas' paper
        # plt.imshow(im, vmin=0, alpha=np.float32(im != 0))
        # ax.axis("off")
    # breakpoint()
    img=np.mean(img_coll,axis=0)
    plt.imshow(img, alpha=np.float32(img != 0))
    plt.axis("off")
    plt.savefig(output_folder+f"/{f_hname}_attention_map_layer_{transformer_layer}.png",dpi=1200)
    # Image.fromarray(img).convert('RGB').save(output_folder+f"/{f_hname}_attention_map_layer_{transformer_layer}.jpg")