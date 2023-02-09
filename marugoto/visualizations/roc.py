#!/usr/bin/env python3

from collections import namedtuple
from typing import Iterable, Sequence, Optional, Tuple, Mapping, List
from pathlib import Path
import argparse
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from warnings import warn

all = [
    "plot_roc_curve",
    "plot_roc_curves",
    "plot_rocs_for_subtypes",
    "plot_roc_curves_",
]


def plot_roc_curve(
    ax: plt.Axes,
    y_true: Sequence[int],
    y_pred: Sequence[float],
    *,
    title: Optional[str] = None,
    label: Optional[str] = None,
) -> int:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.

    Returns:
        The area under the curve.
    """
    warn(
        "ROC curves in marugoto are deprecated.  "
        "Please use wanshi <https://github.com/KatherLab/wanshi-utils/tree/main/wanshi/visualizations> instead.",
        FutureWarning,
    )
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ax.plot(fpr, tpr, label=label)

    style_auc(ax)

    auc = roc_auc_score(y_true, y_pred)
    if title:
        ax.set_title(f"{title}\n(AUC = {auc:0.2f})")
    else:
        ax.set_title(f"AUC = {auc:0.2f}")

    return auc


def style_auc(ax):
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")


TPA = namedtuple("TPA", ["true", "pred", "auc"])


def plot_roc_curves(
    ax: plt.Axes,
    y_trues: Sequence[Sequence[int]],
    y_preds: Sequence[Sequence[float]],
    *,
    title: Optional[str] = None,
) -> Tuple[float, float]:
    """Plots a family of ROC curves.

    Args:
        ax:  Axis to plot to.
        y_trues:  Sequence of ground truth lists.
        y_preds:  Sequence of prediction lists.
        title:  Title of the plot.

    Returns:
        The 95% confidence interval of the area under the curve.
    """
    warn(
        "ROC curves in marugoto are deprecated.  "
        "Please use wanshi <https://github.com/KatherLab/wanshi-utils/tree/main/wanshi/visualizations> instead.",
        FutureWarning,
    )
    # sort trues, preds, AUCs by AUC
    tpas = [
        TPA(t, p, roc_auc_score(t, p)) for t, p in zip(y_trues, y_preds)
    ]  # , strict=True)]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    # plot rocs
    for t, p, auc in tpas:
        fpr, tpr, _ = roc_curve(t, p)
        ax.plot(fpr, tpr, label=f"AUC = {auc:0.2f}")

    # style plot
    style_auc(ax)
    ax.legend()

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    l, h = st.t.interval(0.95, len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))
    conf_range = (h - l) / 2
    auc_str = f"AUC = ${np.mean(aucs):0.2f} \pm {conf_range:0.2f}$"

    if title:
        ax.set_title(f"{title}\n({auc_str})")
    else:
        ax.set_title(auc_str)

    return l, h


def plot_rocs_for_subtypes(
    ax: plt.Axes,
    groups: Mapping[str, Tuple[Sequence[int], Sequence[float]]],
    *,
    target_label: str,
    subgroup_label: str,
    subgroups: Optional[Sequence[str]],
) -> None:
    """Plots a ROC for multiple groups.

    Will a ROC curve for each y_true, y_pred pair in groups, titled with its
    key.  The subg
    """
    warn(
        "ROC curves in marugoto are deprecated.  "
        "Please use wanshi <https://github.com/KatherLab/wanshi-utils/tree/main/wanshi/visualizations> instead.",
        FutureWarning,
    )
    tpas: List[Tuple[str, TPA]] = []
    for subgroup, (y_true, y_pred) in groups.items():
        if subgroups and subgroup not in subgroups:
            continue

        if len(np.unique(y_true)) <= 1:
            print(
                f"subgroup {subgroup} does only have samples of one class... skipping"
            )
            continue

        tpas.append((subgroup, TPA(y_true, y_pred, roc_auc_score(y_true, y_pred))))

    # sort trues, preds, AUCs by AUC
    tpas = sorted(tpas, key=lambda x: x[1].auc, reverse=True)

    # plot rocs
    for subgroup, (t, p, auc) in tpas:
        fpr, tpr, _ = roc_curve(t, p)
        ax.plot(fpr, tpr, label=f"{subgroup} (AUC = {auc:0.2f})")

    # style plot
    style_auc(ax)
    ax.legend(loc="lower right")
    return ax.set_title(f"{target_label} Subgrouped by {subgroup_label}")


def plot_roc_curves_(
    pred_csvs: Iterable[str],
    target_label: str,
    true_label: str,
    outpath: Path,
    subgroup_label: Optional[str],
    subgroups: Optional[Sequence[str]],
    clini_table: Optional[str],
) -> None:
    """Creates ROC curves.

    Args:
        pred_csvs:  A list of prediction CSV files.
        target_label:  The target label to calculate the ROC for.
        true_label:  The positive class for the ROC.
        outpath:  Path to save the `.svg` to.
    """
    warn(
        "ROC curves in marugoto are deprecated.  "
        "Please use wanshi <https://github.com/KatherLab/wanshi-utils/tree/main/wanshi/visualizations> instead.",
        FutureWarning,
    )
    import pandas as pd

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    # prediction dataframes (in 5-fold crossval = 5 dataframes)
    pred_dfs = [pd.read_csv(p, dtype=str) for p in pred_csvs]

    if subgroup_label:
        assert (
            len(pred_dfs) == 1
        ), "currently subgroup analysis is only supported for a singular set of predictions"
        pred_df = pred_dfs[0]

        # get clini table for extracting subgroups from respective column
        clini = pd.read_excel(clini_table)

        groups = {}
        for subgroup, subgroup_patients in clini.PATIENT.groupby(clini[subgroup_label]):
            subgroup_preds = pred_df[pred_df.PATIENT.isin(subgroup_patients)]
            y_true = subgroup_preds[target_label] == true_label
            y_pred = pd.to_numeric(subgroup_preds[f"{target_label}_{true_label}"])
            groups[subgroup] = (y_true, y_pred)

        fig, ax = plt.subplots()
        plot_rocs_for_subtypes(
            ax,
            groups,
            target_label=target_label,
            subgroup_label=subgroup_label,
            subgroups=subgroups,
        )
        fig.savefig(outpath / f"roc-{target_label}-subtyped_by_{subgroup_label}.svg")

    else:
        y_trues = [df[target_label] == true_label for df in pred_dfs]
        y_preds = [pd.to_numeric(df[f"{target_label}_{true_label}"]) for df in pred_dfs]
        title = f"{target_label} = {true_label}"

        fig, ax = plt.subplots()
        if len(pred_dfs) == 1:
            plot_roc_curve(ax, y_trues[0], y_preds[0], title=title)
        else:
            plot_roc_curves(ax, y_trues, y_preds, title=title)
        fig.savefig(outpath / f"roc-{target_label}={true_label}.svg")
        plt.close(fig)


if __name__ == "__main__":
    warn(
        "ROC curves in marugoto are deprecated.  "
        "Please use wanshi <https://github.com/KatherLab/wanshi-utils/tree/main/wanshi/visualizations> instead.",
        FutureWarning,
    )
    parser = argparse.ArgumentParser(description="Create a ROC Curve.")
    parser.add_argument(
        "pred_csvs",
        metavar="PREDS_CSV",
        nargs="+",
        type=Path,
        help="Predictions to create ROC curves for.",
    )
    parser.add_argument(
        "--target-label",
        required=True,
        type=str,
        help="The target label to calculate the ROC for.",
    )
    parser.add_argument(
        "--true-label",
        required=True,
        type=str,
        help="The target label to calculate the ROC for.",
    )
    parser.add_argument(
        "-o", "--outpath", required=True, type=Path, help="Path to save the `.svg` to."
    )
    parser.add_argument(
        "--subgroup-label",
        required=False,
        type=str,
        help="Column name in Clini where to get the subgroups from.",
    )
    parser.add_argument(
        "--subgroup",
        metavar="SUBGROUP",
        dest="subgroups",
        required=False,
        type=str,
        action="append",
        help=(
            "A subgroup to include in the output.  "
            "If none are given, a ROC curve for each of the subgroups will be created."
        ),
    )
    parser.add_argument(
        "--clini-table",
        required=False,
        type=Path,
        help="Path to get subgroup information from Clini table from.",
    )
    args = parser.parse_args()
    if args.clini_table is not None and args.subgroup_label is None:
        parser.error(
            "supplying a clini table only makes sense if `--subgroup-label` is specified"
        )

    plot_roc_curves_(**vars(args))
