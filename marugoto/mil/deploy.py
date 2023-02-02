from argparse import ArgumentParser
from pathlib import Path

from fastai.vision.learner import load_learner

from marugoto.mil._mil import deploy
from marugoto.mil.data import get_cohort_df, get_target_enc


def add_deploy_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds arguments required for model deployment to an ArgumentParser."""
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
        "--model",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path of the model to deploy.",
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        type=str,
        required=False,
        help="Label to train for. Inferred from model, if none given.",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Path to write the outputs to.",
    )

    return parser


if __name__ == "__main__":
    parser = ArgumentParser("Deploy a categorical model.")
    parser = add_deploy_args(parser)
    args = parser.parse_args()

    if (preds_csv := args.output_dir / "patient-preds.csv").exists():
        print(f"{preds_csv} already exists!  Skipping...")
        exit(0)

    learn = load_learner(args.model_path)
    target_enc = get_target_enc(learn)

    categories = target_enc.categories_[0]

    target_label = args.target_label or learn.target_label

    test_df = get_cohort_df(
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        feature_dir=args.feature_dir,
        target_label=target_label,
        categories=categories,
    )
    patient_preds_df = deploy(test_df=test_df, learn=learn, target_label=target_label)
    args.output_path.mkdir(parents=True, exist_ok=True)
    patient_preds_df.to_csv(preds_csv, index=False)
