Note: Requires Python 3.8. For more information regarding setting up marugoto on your system and file management for your project, see [documentation](https://github.com/KatherLab/marugoto/blob/main/Documentation.md).

# まるごと—A Toolbox for Building Deep Learning Workflows ##
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

## Attention-Based Multiple Instance Learning ##
### Cross-validate a model on pre-extracted features ###

    python -m marugoto.mil crossval \
        --clini-table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path \
        --n-splits 5

### Train a single model on pre-extracted features ###

    python -m marugoto.mil train \
        --clini-table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target-label isMSIH \
        --output-path output/path

### Deploy a model on pre-extracted features from another cohort ###

    python -m marugoto.mil deploy \
        --clini_table tcga-crc-dx/TCGA-CRC-DX_CLINI.xlsx \
        --slide-csv tcga-crc-dx/TCGA-CRC-DX_SLIDE.csv \
        --feature-dir tcga-crc-dx/features_norm_macenko_h5 \
        --target_label isMSIH \
        --model-path training-dir/export.pkl \
        --output_path output/path


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
