# まるごと—A Toolbox for Building Deep Learning Workflows ##

Note: for more information regarding setting up marugoto on your system and file management for your project, see [documentation](https://github.com/KatherLab/marugoto/blob/main/Documentation.md).


Our workflow uses a DL approach which performs self-supervised feature extraction with attMIL workflow. This approach addresses a weakly supervised classification problem in which the objective is to predict a slide label from a collection of individual tiles.
Some studies have reported a performance gain of the self-supervised-learning attMIL approach compared to the classical approach.
Wang et al. trained a ResNet-50 on 32000 WSIs from TCGA via the RetCCL self-supervised learning algorithm. We used pre-trained architectures to extract 2048 features (“Wang-attMIL”) per tile.

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
        --clini_table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path \
        (optional/recommended especially during training) --tile_no 256

### Deploy a Model on Another Cohort ###

    python -m marugoto.features deploy \
        --clini_table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --model-path training-dir/export.pkl \
        --output-path output/path \
        (optional) --tile_no 256

### Cross-Validate a Model ###

    python -m marugoto.features crossval \
        --clini_table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path \
        --n-splits 5 \
        (optional) --fixed_folds /abs/path/to/folds.pt

## Attention-Based Multiple Instance Learning ##

### Train a Neural Network on Pre-Extracted Features ###

    python -m marugoto.mil train \
        --clini-table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path

### Deploy a Model on Another Cohort ###

    python -m marugoto.mil deploy \
        --clini_table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target_label isMSIH \
        --model-path training-dir/export.pkl \
        --output_path output/path

### Cross-Validate a Model ###

    python -m marugoto.mil crossval \
        --clini-table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path \
        --n-splits 5

## Calculate Statistics for Categorical Deployments ##

    python -m marugoto.stats.categorical \
        deployment/path/fold-*/patient-preds.csv \
        --outpath output/path \
        --target_label isMSIH

## Plot ROC Curve ##

    python -m marugoto.visualizations.roc \
        deployment/path/fold-*/patient-preds.csv \
        --outpath output/path \
        --target-label isMSIH \
        --true-label MSIH \
        --clini-table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \ (optional: subgroup analysis) 
        --subgroup-label 'PRETREATED' (optional: subgroup analysis) 

## Plot Precision Recall Curve ##

    python -m marugoto.visualizations.prc \
        deployment/path/fold-*/patient-preds.csv \
        --outpath output/path \
        --target-label isMSIH \
        --true-label MSIH
        

## Advice

- default batch size is 64 patients, consider to adapt for smaller cohorts


## Running Marugoto in a Container

Marugoto can be conveniently run in a podman container.  To do so, use the
`marugoto-container.sh` convenience script.  Training a MIL model can be done as
follows:

```sh
./marugoto-container.sh \
    marugoto.mil train \
        --clini-table /workdir/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv /workdir/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir /workdir/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path /results
```

For more information on how to run podman containers, please refer to the podman
documentation.
