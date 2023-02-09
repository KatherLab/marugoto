from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from fastai.vision.learner import load_learner
import torch

from ._features import train, deploy

__all__ = [
    "train_categorical_model_",
    "deploy_categorical_model_",
    "categorical_crossval_",
]


PathLike = Union[str, Path]


def train_categorical_model_(
    clini_table: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    target_label: str,
    output_path: PathLike,
    categories: Optional[Iterable[str]] = None,
    tile_no=None,
) -> None:
    """Train a categorical model on a cohort's tile's features.

    Args:
        clini_table:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
        feature_dir:  Path containing the features.
        output_path:  File to save model in.
    """
    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    from datetime import datetime

    info: Dict[str, Any] = {
        "description": "training on tile features",
        "clini": str(Path(clini_table).absolute()),
        "slide": str(Path(slide_csv).absolute()),
        "feature_dir": str(feature_dir.absolute()),
        "target_label": str(target_label),
        "output_path": str(output_path.absolute()),
        "datetime": datetime.now().astimezone().isoformat(),
    }

    model_path = output_path / "export.pkl"
    if model_path.exists():
        print(f"{model_path} already exists. Skipping...")
        return

    clini_df = (
        pd.read_csv(clini_table, dtype=str)
        if Path(clini_table).suffix == ".csv"
        else pd.read_excel(clini_table, dtype=str)
    )
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on="PATIENT")

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    if not categories:
        categories = df[target_label].unique()
        print(f"Inferred {categories = }")
    categories = np.array(categories)
    info["categories"] = list(categories)

    df = df[df[target_label].isin(categories)]

    slides = set(feature_dir.glob("*.h5"))
    # remove slides we don't have
    slide_df = pd.DataFrame(slides, columns=["slide_path"])
    slide_df["FILENAME"] = slide_df.slide_path.map(lambda p: p.stem)
    df = df.merge(slide_df, on="FILENAME")

    print("Overall distribution")
    print(df[target_label].value_counts())
    info["class distribution"] = {
        "overall": {k: int(v) for k, v in df[target_label].value_counts().items()}
    }

    # Split off validation set
    patient_df = df.groupby("PATIENT").first()

    train_patients, valid_patients = train_test_split(
        patient_df.index, stratify=patient_df[target_label]
    )
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]

    info["class distribution"]["training"] = {
        k: int(v) for k, v in train_df[target_label].value_counts().items()
    }
    info["class distribution"]["validation"] = {
        k: int(v) for k, v in valid_df[target_label].value_counts().items()
    }

    target_enc = OneHotEncoder(sparse=False).fit(categories.reshape(-1, 1))

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f)
    
    learn, patient_preds_df, tile_scores_df = train(
        target_enc=target_enc,
        train_bags=train_df.slide_path.values,
        train_targets=train_df[target_label].values,
        valid_bags=valid_df.slide_path.values,
        valid_targets=valid_df[target_label].values,
        valid_df=valid_df,
        target_label=target_label,
        path=output_path,
        tile_no=tile_no,
    )
  
    learn.export()
    patient_preds_df.to_csv(output_path / "patient-preds-validset.csv")
    tile_scores_df.to_csv(output_path / "tile-preds-validset.csv")


def deploy_categorical_model_(
    clini_table: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    target_label: str,
    model_path: PathLike,
    output_path: PathLike,
    tile_no: int = None,
) -> None:
    """Deploy a categorical model on a cohort's tile's features.

    Args:
        clini_table:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        feature_dir:  Path containing the features.
        model_path:  Path of the model to deploy.
        output_path:  File to save model in.
    """
    feature_dir = Path(feature_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if (preds_csv := output_path / "patient-preds.csv").exists():
        print(f"{preds_csv} already exists!  Skipping...")
        return

    clini_df = (
        pd.read_csv(clini_table, dtype=str)
        if Path(clini_table).suffix == ".csv"
        else pd.read_excel(clini_table, dtype=str)
    )
    slide_df = pd.read_csv(slide_csv, dtype=str)
    test_df = clini_df.merge(slide_df, on="PATIENT")

    learn = load_learner(model_path)
    target_enc = learn.dls.train.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]

    # remove uninteresting
    test_df = test_df[test_df[target_label].isin(categories)]
    # remove slides we don't have
    slides = set(feature_dir.glob("*.h5"))
    slide_df = pd.DataFrame(slides, columns=["slide_path"])
    slide_df["FILENAME"] = slide_df.slide_path.map(lambda p: p.stem)
    test_df = test_df.merge(slide_df, on="FILENAME")
    
    patient_preds_df, tile_preds_df = deploy(
        test_df=test_df, learn=learn, target_label=target_label, tile_no=tile_no
    )

    patient_preds_df.to_csv(preds_csv, index=False)
    tile_preds_df.to_csv(output_path / "tile-preds.csv", index=False)

def categorical_crossval_(
    clini_excel: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    target_label: str,
    output_path: PathLike,
    n_splits: int = 5,
    categories=None,
) -> None:
    """Performs a cross-validation for a categorical target.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        feature_dir:  Path containing the features.
        target_label:  Label to train for.
        output_path:  File to save model and the results in.
        n_splits:  The number of folds.
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
    """
    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    info = {
        "description": "cross-validation on tile features",
        "clini": str(Path(clini_excel).absolute()),
        "slide": str(Path(slide_csv).absolute()),
        "feature_dir": str(feature_dir.absolute()),
        "target_label": str(target_label),
        "output_path": str(output_path.absolute()),
        "n_splits": n_splits,
        "datetime": datetime.now().astimezone().isoformat(),
    }

    clini_df = pd.read_excel(clini_excel, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on="PATIENT")

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    if not categories:
        categories = df[target_label].unique()
        print(f"Inferred {categories = }")
    categories = np.array(categories)
    info["categories"] = list(categories)

    df = df[df[target_label].isin(categories)]

    slides = set(feature_dir.glob("*.h5"))
    # remove slides we don't have
    slide_df = pd.DataFrame(slides, columns=["slide_path"])
    slide_df["FILENAME"] = slide_df.slide_path.map(lambda p: p.stem)
    df = df.merge(slide_df, on="FILENAME")

    info["class distribution"] = {
        "overall": {k: int(v) for k, v in df[target_label].value_counts().items()}
    }

    target_enc = OneHotEncoder(sparse=False).fit(categories.reshape(-1, 1))

    if (fold_path := output_path / "folds.pt").exists():
        folds = torch.load(fold_path)
    else:
        skf = StratifiedKFold(n_splits=n_splits)
        patient_df = df.groupby("PATIENT").first().reset_index()
        folds = tuple(skf.split(patient_df.PATIENT, patient_df[target_label]))
        torch.save(folds, fold_path)

    info["folds"] = [
        {
            part: list(df.PATIENT[folds[fold][i]])
            for i, part in enumerate(["train", "test"])
        }
        for fold in range(n_splits)
    ]

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f)

    for fold, (train_idxs, test_idxs) in enumerate(folds):
        fold_path = output_path / f"fold-{fold}"
        if (preds_csv := fold_path / "patient-preds.csv").exists():
            print(f"{preds_csv} already exists!  Skipping...")
            continue
        elif (fold_path / "export.pkl").exists():
            learn = load_learner(fold_path / "export.pkl")
        else:
            fold_train_df = df.iloc[train_idxs]
            learn = _crossval_train(
                fold_path=fold_path,
                fold_df=fold_train_df,
                fold=fold,
                info=info,
                target_label=target_label,
                target_enc=target_enc,
            )
            learn.export()

        from marugoto.features import deploy

        fold_test_df = df.iloc[test_idxs]
        patient_preds_df = deploy(
            test_df=fold_test_df, learn=learn, target_label=target_label
        )
        patient_preds_df.to_csv(preds_csv, index=False)


def _crossval_train(*, fold_path, fold_df, fold, info, target_label, target_enc):
    """Helper function for training the folds."""
    fold_path.mkdir(exist_ok=True, parents=True)

    patient_df = fold_df.groupby("PATIENT").first()

    info["class distribution"][f"fold {fold}"] = {
        "overall": {k: int(v) for k, v in fold_df[target_label].value_counts().items()}
    }

    train_patients, valid_patients = train_test_split(
        patient_df.index, stratify=patient_df[target_label]
    )
    train_df = fold_df[fold_df.PATIENT.isin(train_patients)]
    valid_df = fold_df[fold_df.PATIENT.isin(valid_patients)]

    info["class distribution"][f"fold {fold}"]["training"] = {
        k: int(v) for k, v in train_df[target_label].value_counts().items()
    }
    info["class distribution"][f"fold {fold}"]["validation"] = {
        k: int(v) for k, v in valid_df[target_label].value_counts().items()
    }

    from marugoto.features import train

    learn = train(
        target_enc=target_enc,
        train_bags=train_df.slide_path.values,
        train_targets=train_df[target_label].values,
        valid_bags=valid_df.slide_path.values,
        valid_targets=valid_df[target_label].values,
        path=fold_path,
    )

    return learn
