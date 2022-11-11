from datetime import datetime
import json
from pathlib import Path
from pyexpat import features
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from fastai.vision.learner import load_learner
import torch
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import scipy
import os
#from marugoto.data import FunctionTransformer

from ._mil import train, deploy
from .data import get_cohort_df, get_target_enc

__all__ = [
    'train_categorical_model_', 'deploy_categorical_model_', 'categorical_crossval_']


PathLike = Union[str, Path]


def train_categorical_model_(
    clini_table: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    output_path: PathLike,
    *,
    target_label: str,
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    categories: Optional[Iterable[str]] = None,
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



    model_path = output_path/'export.pkl'
    if model_path.exists():
        print(f'{model_path} already exists. Skipping...')
        return

    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    # TODO move into get_cohort_df
    if not categories:
        categories = df[target_label].unique()
    categories = np.array(categories)
    # info['categories'] = list(categories)


    #NOTE: HERE THE TARGETS ARE MIN-MAX NORMALIZED COLUMN WISE
    df = get_cohort_df(clini_table, slide_csv, feature_dir, target_label)
    scaler=MinMaxScaler()

    df[target_label] = scaler.fit_transform(df[[target_label]])


    # Split off validation set
    train_patients, valid_patients = train_test_split(df.PATIENT) #, stratify=df[target_label]
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]
    train_df.drop(columns='slide_path').to_csv(output_path/'train.csv', index=False)
    valid_df.drop(columns='slide_path').to_csv(output_path/'valid.csv', index=False)

    add_features = []
    if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), df[cat_labels].values))
    if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), df[cont_labels].values))

    learn = train(
        bags=df.slide_path.values,
        targets=df[target_label].values,
        add_features=add_features,
        valid_idxs=df.PATIENT.isin(valid_patients),
        path=output_path,
    )

    # save some additional information to the learner to make deployment easier
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    learn.export()


def _make_cat_enc(df, cats) -> FunctionTransformer:
    # create a scaled one-hot encoder for the categorical values
    #
    # due to weirdeties in sklearn's OneHotEncoder.fit we fill NAs with other values
    # randomly sampled with the same probability as their distribution in the
    # dataset.  This is necessary for correctly determining StandardScaler's weigth
    fitting_cats = []
    for cat in cats:
        weights = df[cat].value_counts(normalize=True)
        non_na_samples = df[cat].fillna(pd.Series(np.random.choice(weights.index, len(df), p=weights)))
        fitting_cats.append(non_na_samples)
    cat_samples = np.stack(fitting_cats, axis=1)
    cat_enc = make_pipeline(
        FunctionTransformer(), #OneHotEncoder(sparse=False, handle_unknown='ignore'),
        StandardScaler(),
    ).fit(cat_samples)
    return cat_enc


def _make_cont_enc(df, conts) -> FunctionTransformer:
    cont_enc = make_pipeline(
        StandardScaler(),
        SimpleImputer(fill_value=0)
    ).fit(df[conts].values)
    return cont_enc


def deploy_categorical_model_(
    clini_table: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    model_path: PathLike,
    output_path: PathLike,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[str] = None,
    cont_labels: Optional[str] = None,
) -> None:
    """Deploy a categorical model on a cohort's tile's features.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        feature_dir:  Path containing the features.
        model_path:  Path of the model to deploy.
        output_path:  File to save model in.
    """
    feature_dir = Path(feature_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)
    if (preds_csv := output_path/'patient-preds.csv').exists():
        print(f'{preds_csv} already exists!  Skipping...')
        return

    learn = load_learner(model_path)


    test_df = get_cohort_df(clini_table, slide_csv, feature_dir, target_label) #, categories
    print(f'Initial size: {test_df.shape}')
    test_df = test_df.drop((test_df[test_df[target_label].isna()]).index)
    print(f'Post-NA removal size: {test_df.shape}')
    # scaler=MinMaxScaler()
    # test_df[target_label] = scaler.fit_transform(test_df[[target_label]])

    #saving test.csv for later ROC curve generation

    patient_preds_df = deploy(test_df=test_df, learn=learn, target_label=target_label)
    output_path.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(f'{output_path}/test.csv', index=False)
    patient_preds_df.to_csv(preds_csv, index=False)


def categorical_crossval_(
    clini_excel: PathLike, slide_csv: PathLike, feature_dir: PathLike, output_path: PathLike,
    *,
    target_label: str,
    binary_label: Optional[str] = None,
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    n_splits: int = 5,
    categories: Optional[Iterable[str]] = None,
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
        'description': 'MIL cross-validation',
        'clini': str(Path(clini_excel).absolute()),
        'slide': str(Path(slide_csv).absolute()),
        'feature_dir': str(feature_dir.absolute()),
        'target_label': str(target_label),
        'cat_labels': [str(c) for c in cat_labels],
        'cont_labels': [str(c) for c in cont_labels],
        'output_path': str(output_path.absolute()),
        'n_splits': n_splits,
        'datetime': datetime.now().astimezone().isoformat()}

    clini_df = pd.read_csv(clini_excel, dtype=str) if Path(clini_excel).suffix == '.csv' else pd.read_excel(clini_excel, dtype=str)
    clini_df = clini_df.astype({target_label: 'float32'})
    
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    #CHANGED
    df = get_cohort_df(clini_excel, slide_csv, feature_dir, target_label) #categories
    
    #get rid of NA values in the target for the clini table
    df = (df.drop((df[df[target_label].isna()]).index)).reset_index()

    if (fold_path := output_path/'folds.pt').exists():
        folds = torch.load(fold_path)
    else:
        #added shuffling with seed 1337
        if binary_label is None:
            skf = KFold(n_splits=n_splits, shuffle=True, random_state=1337)
            patient_df = df.groupby('PATIENT').first().reset_index()
            folds = tuple(skf.split(patient_df.PATIENT, patient_df[target_label])) # patient_df['SITE_CODE'])) with stratified potentially
            torch.save(folds, fold_path)
        #add option to create balanced folds based on binary equivalent
        else:
            print(f"Using StratifiedKFold with binarized variable {binary_label}")
            all_classes = df[binary_label].unique()
            least_populated_class = min([np.sum(df[binary_label] == x) for x in all_classes])
            if least_populated_class < n_splits:
                print(f"Warning: Cannot make requested {n_splits} folds, reduced to {least_populated_class} folds.")
                n_splits = least_populated_class
                info['n_splits'] = least_populated_class
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
            patient_df = df.groupby('PATIENT').first().reset_index()
            folds = tuple(skf.split(patient_df.PATIENT, patient_df[binary_label])) # patient_df['SITE_CODE'])) with stratified potentially
            torch.save(folds, fold_path)          

    info['folds'] = [
        {
            part: list(df.PATIENT[folds[fold][i]])
            for i, part in enumerate(['train', 'test'])
        }
        for fold in range(info['n_splits']) ]

    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)

    for fold, (train_idxs, test_idxs) in enumerate(folds):
        fold_path = output_path/f'fold-{fold}'
        
        #minmax normalisation for train set, save distrib for test
        fold_train_df = pd.DataFrame(df.iloc[train_idxs])
        scaler=MinMaxScaler().fit(fold_train_df[target_label].values.reshape(-1,1))
        fold_train_df[target_label] = scaler.transform(fold_train_df[target_label].values.reshape(-1,1))

        if (preds_csv := fold_path/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            continue
        elif (fold_path/'export.pkl').exists():
            learn = load_learner(fold_path/'export.pkl')
        else:         
            learn = _crossval_train(
                fold_path=fold_path, fold_df=fold_train_df, fold=fold, info=info,
                target_label=target_label, #, target_enc=target_enc,
                cat_labels=cat_labels, cont_labels=cont_labels, binary_label=binary_label) #added weights #fold_weights_train=fold_weights_train
            learn.export()

        #minmax normalisation for test set with train distrib (same scaler object)
        fold_test_df = pd.DataFrame(df.iloc[test_idxs])
        fold_test_df.drop(columns='slide_path').to_csv(fold_path/'test.csv', index=False)
        fold_test_df[target_label] = scaler.transform(fold_test_df[target_label].values.reshape(-1,1))
        
        patient_preds_df = deploy(
            test_df=fold_test_df, learn=learn, #send weights to be all ones, i.e. nothing changes weights=np.ones(test_idxs.shape)
            target_label=target_label, cat_labels=cat_labels, cont_labels=cont_labels)

        #rescale ground truth and patient predictions to original range
        patient_preds_df[target_label] = scaler.inverse_transform(patient_preds_df[target_label].values.reshape(-1,1))
        patient_preds_df['pred'] = scaler.inverse_transform(patient_preds_df['pred'].values.reshape(-1,1))

        #obtain pearson's R and create plot per fold
        plot_pearsr_df = patient_preds_df[[target_label, "pred"]]
        pears = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[0]
        pval = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[1]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plot_pearsr_df[target_label], plot_pearsr_df['pred'])
        ax = sns.lmplot(x=target_label, y='pred', data=plot_pearsr_df)
        ax.set(title=f"{os.path.basename(output_path)}\nR^2: {np.round(r_value**2, 2)} | Pearson's R: {np.round(pears,2)} | p-value: {np.round(pval, 7)}")
        #ax.set(ylim=(0,1), xlim=(0,1)) #set a x/y-limit to get the same plots for a specific project
        ax.savefig(fold_path/"correlation_plot.png")

        patient_preds_df.to_csv(preds_csv, index=False)
    ######

    #CHANGED
def _crossval_train(
    *, fold_path, fold_df, fold, info, target_label, cat_labels, cont_labels, binary_label #target_enc,fold_weights_train 
):
    """Helper function for training the folds."""
    assert fold_df.PATIENT.nunique() == len(fold_df)
    fold_path.mkdir(exist_ok=True, parents=True)

    #CHANGED
    #added stratification at train_test_split
    if binary_label is not None:
        train_patients, valid_patients = train_test_split(
            fold_df.PATIENT, stratify=fold_df[binary_label])
    else:
        train_patients, valid_patients = train_test_split(
            fold_df.PATIENT)
    train_df = fold_df[fold_df.PATIENT.isin(train_patients)]
    valid_df = fold_df[fold_df.PATIENT.isin(valid_patients)]
    train_df.drop(columns='slide_path').to_csv(fold_path/'train.csv', index=False)
    valid_df.drop(columns='slide_path').to_csv(fold_path/'valid.csv', index=False)

    add_features = []
    if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), fold_df[cat_labels].values))
    if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), fold_df[cont_labels].values))


    #CHANGED
    #all inputs for the train() function seem to be what we want. Reshaped the targets to match the Tensor size [64,1] and [1,1] 
    learn = train(
        bags=fold_df.slide_path.values,
        targets=(fold_df[target_label].values).reshape(-1,1), #(373,1) enters here, i.e. ALL target data
        add_features=add_features,
        valid_idxs=fold_df.PATIENT.isin(valid_patients),
        path=fold_path) 
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    return learn
