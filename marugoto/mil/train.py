#!/usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
import torch.nn.functional as F

from marugoto.mil._mil import train
from marugoto.mil.data import get_cohort_df
from marugoto.mil.helpers import _make_cat_enc, _make_cont_enc


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds arguments required for model training to an ArgumentParser."""
    parser.add_argument(
        "--clini-table",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path of the clini table.",
    )
    parser.add_argument(
        "--slide-table",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path of the slide table.",
    )
    parser.add_argument(
        "--feature-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Path the h5 features are saved in.",
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        type=str,
        required=True,
        help="Label to train for.",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Path to write the outputs to.",
    )
    parser.add_argument(
        "--category",
        metavar="CAT",
        dest="categories",
        type=str,
        required=False,
        action="append",
        help=(
            "Category to train for. "
            "Has to be an entry specified in the clini table's column "
            "specified by `--clini-table`. "
            "Can be given multiple times to specify multiple categories to train for. "
        ),
    )

    multimodal_group = parser.add_argument_group("Multimodal training")
    multimodal_group.add_argument(
        "--additional-training-category",
        metavar="LABEL",
        dest="cat_labels",
        type=str,
        required=False,
        action="append",
        help="Categorical column in clini table to additionally use in training.",
    )
    multimodal_group.add_argument(
        "--additional-training-continuous",
        metavar="LABEL",
        dest="cont_labels",
        type=str,
        required=False,
        action="append",
        help="Continuous column in clini table to additionally use in training.",
    )

    return parser


if __name__ == "__main__":
    parser = ArgumentParser("Train a categorical model on a cohort's tile's features.")
    add_train_args(parser)
    args = parser.parse_args()
    print(args)

    args.output_dir.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    info: Dict[str, Any] = {
        "description": "MIL training",
        "clini": str(args.clini_table.absolute()),
        "slide": str(args.slide_table.absolute()),
        "feature_dir": str(args.feature_dir.absolute()),
        "target_label": str(args.target_label),
        "cat_labels": [str(c) for c in (args.cat_labels or [])],
        "cont_labels": [str(c) for c in (args.cont_labels or [])],
        "output_dir": str(args.output_dir.absolute()),
        "datetime": datetime.now().astimezone().isoformat(),
    }

    model_path = args.output_dir / "export.pkl"
    if model_path.exists():
        print(f"{model_path} already exists. Skipping...")
        exit(0)

    df, categories = get_cohort_df(
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        feature_dir=args.feature_dir,
        target_label=args.target_label,
        categories=args.categories,
    )

    print("Overall distribution")
    print(df[args.target_label].value_counts())
    assert not df[
        args.target_label
    ].empty, "no input dataset. Do the tables / feature dir belong to the same cohorts?"

    info["class distribution"] = {
        "overall": {k: int(v) for k, v in df[args.target_label].value_counts().items()}
    }

    # Split off validation set
    train_patients, valid_patients = train_test_split(
        df.PATIENT, stratify=df[args.target_label]
    )
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]
    train_df.drop(columns="slide_path").to_csv(
        args.output_dir / "train.csv", index=False
    )
    valid_df.drop(columns="slide_path").to_csv(
        args.output_dir / "valid.csv", index=False
    )

    info["class distribution"]["training"] = {
        k: int(v) for k, v in train_df[args.target_label].value_counts().items()
    }
    info["class distribution"]["validation"] = {
        k: int(v) for k, v in valid_df[args.target_label].value_counts().items()
    }

    with open(args.output_dir / "info.json", "w") as f:
        json.dump(info, f)

    target_enc = OneHotEncoder(sparse=False).fit(categories.reshape(-1, 1))

    add_features = []
    if args.cat_labels:
        add_features.append(
            (_make_cat_enc(train_df, args.cat_labels), df[args.cat_labels].values)
        )
    if args.cont_labels:
        add_features.append(
            (_make_cont_enc(train_df, args.cont_labels), df[args.cont_labels].values)
        )

    learn = train(
        bags=df.slide_path.values,
        targets=(target_enc, df[args.target_label].values),
        add_features=add_features,
        valid_idxs=df.PATIENT.isin(valid_patients).values,
        path=args.output_dir,
    )

    # save some additional information to the learner to make deployment easier
    learn.target_label = args.target_label
    learn.cat_labels, learn.cont_labels = args.cat_labels, args.cont_labels

    learn.export()

    patient_preds, patient_targs = learn.get_preds(act=nn.Softmax(dim=1))

    patient_preds_df = pd.DataFrame.from_dict(
        {
            "PATIENT": valid_df.PATIENT.values,
            args.target_label: valid_df[args.target_label].values,
            **{
                f"{args.target_label}_{cat}": patient_preds[:, i]
                for i, cat in enumerate(categories)
            },
        }
    )

    # calculate loss
    patient_preds = patient_preds_df[
        [f"{args.target_label}_{cat}" for cat in categories]
    ].values
    patient_targs = target_enc.transform(
        patient_preds_df[args.target_label].values.reshape(-1, 1)
    )
    patient_preds_df["loss"] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs), reduction="none"
    )

    patient_preds_df["pred"] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[
        [
            "PATIENT",
            args.target_label,
            "pred",
            *(f"{args.target_label}_{cat}" for cat in categories),
            "loss",
        ]
    ]
    patient_preds_df = patient_preds_df.sort_values(by="loss")
    patient_preds_df.to_csv(args.output_dir / "patient-preds-validset.csv", index=False)
