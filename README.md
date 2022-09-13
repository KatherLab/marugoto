# Regression まるごと—A Toolbox for Building Deep Learning Regression Workflows ##

This repo has been adapted from the main Marugoto pipeline which only allowed classification problems.
Please contact omarelnahhas1337@gmail.com to get it running in your environment, as packages such as sklearn have been changed in order to make it work, which have not been pushed to this repo.


Note: for more information regarding setting up marugoto on your system and file management for your project, see [documentation](https://github.com/KatherLab/marugoto/blob/main/Documentation.md).

## Tile-Wise Training and Deployment ##

### Train a Neural Network on Pre-Extracted Features ###

    python -m marugoto.features train \
        --clini-excel tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path

### Deploy a Model on Another Cohort ###

    python -m marugoto.features deploy \
        --clini-excel tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --model-path training-dir/export.pkl \
        --output-path output/path

### Cross-Validate a Model ###

    python -m marugoto.features crossval \
        --clini-excel tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path \
        --n-splits 5

## Attention-Based Multiple Instance Learning ##

### Train a Neural Network on Pre-Extracted Features ###

    python -m marugoto.mil train \
        --clini-excel tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path

### Deploy a Model on Another Cohort ###

    python -m marugoto.mil deploy \
        --clini-excel tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --model-path training-dir/export.pkl \
        --output-path output/path

### Cross-Validate a Model ###

    python -m marugoto.mil crossval \
        --clini-excel tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path \
        --n-splits 5

## Calculate Statistics for Categorical Deployments ##

    python -m marugoto.stats.categorical \
        deployment/path/fold-*/patient-preds.csv \
        --outpath output/path \
        --target-label isMSIH \
        --true-label MSIH

## Plot ROC Curve ##

    python -m marugoto.visualizations.roc  \
        deployment/path/fold-*/patient-preds.csv \
        --outpath output/path \
        --target-label isMSIH \
        --true-label MSIH

## Plot Precision Recall Curve ##

    python -m marugoto.visualizations.prc  \
        deployment/path/fold-*/patient-preds.csv \
        --outpath output/path \
        --target-label isMSIH \
        --true-label MSIH
