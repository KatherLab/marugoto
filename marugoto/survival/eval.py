import os
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loss import cox_loss
from fastai.data.load import DataLoader
from mil.model import MILModel
from mil.data import make_dataset, get_cohort_df
from train import prediction, get_logger


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('-ct', '--clinical_table', type=Path, required=True, help='clinical_table')
parser.add_argument('-st', '--slide_table', type=Path, required=True, help='slide_table')
parser.add_argument('-f', '--feature_dir', type=Path, required=True, help='feature_dir')
parser.add_argument('-o', '--output_path', type=Path, required=True, help='output_path')
parser.add_argument('-m', '--model_path', type=Path, required=True, help='model_path')
parser.add_argument('-t', '--target_label', nargs='+', type=str, required=True, help='target_label, e.g., [os, os_e]')
parser.add_argument('-c', '--cohort', type=str, required=True, help='cohort name')


if __name__ == '__main__':
    args = parser.parse_args()
    clini_excel = args.clinical_table
    slide_csv = args.slide_table
    feature_dir = args.feature_dir
    output_path = args.output_path
    model_path = args.model_path
    target_label = args.target_label
    cohort = args.cohort


    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = get_cohort_df(clini_excel, slide_csv, feature_dir, target_label, categories=None)

    logger = get_logger(output_path / f'{cohort}_eval_exp.log')
    logger.info(f'test:{len(df)}')

    add_features = []

    targs = df[target_label].values
    bags = df.slide_path.values


    test_ds = make_dataset(
        bags=bags,
        # targets=(target_enc, targs[~valid_idxs]),
        targets=targs,
        add_features=[
            (enc, vals)
            for enc, vals in add_features],
        bag_size=None)



    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=os.cpu_count())

    batch = test_dl.one_batch()

    model = MILModel(batch[0].shape[-1], 1) #2048 the len of feature
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    criterion = cox_loss
    with torch.no_grad():
        model.eval()

        test_loss_dict, test_ci_dict, test_score = prediction(model, test_dl, criterion)

    for key in test_ci_dict.keys():
        logger.info(f'test ci_{key}: {test_ci_dict[key]}')


    score = test_score.cpu().detach().numpy()
    logger.info(np.median(score))
    test_df = df.reset_index()
    test_df['SCORE'] = list(score.flatten())

    test_df.to_csv(output_path / f'{cohort}_score.csv')

