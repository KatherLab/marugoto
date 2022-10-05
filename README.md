# まるごと—A Toolbox for Building Deep Learning Workflows ##

Note: for more information regarding setting up marugoto on your system and file management for your project, see [documentation](https://github.com/KatherLab/marugoto/blob/main/Documentation.md).

## Feature extraction
Download best_ckpt.pth from latest Xiyue Wang:
https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL
    
    python -m marugoto.extract.xiyue_wang \
        --checkpoint-path ~/Downloads/best_ckpt.pth \
        --outdir ~/TCGA_features/TCGA-CRC-DX-features/xiyue-wang \
        /mnt/TCGA_BLOCKS/TCGA-CRC-DX-BLOCKS/*
        
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
        --n-splits 5 \
        (optional) --fixed_folds /abs/path/to/folds.pt

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
        

## Advice

- default batch size is 64 patients, consider to adapt for smaller cohorts
