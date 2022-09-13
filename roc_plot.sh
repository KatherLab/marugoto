python -m marugoto.visualizations.roc  \
    /home/omarelnahhas/TCGA_Markowetz/CX1/deploy_FAKE_TCGA-ALL-features_DIR_DIR/fold-*/patient-preds.csv \
    --outpath /home/omarelnahhas/TCGA_Markowetz/CX1/deploy_FAKE_TCGA-ALL-features_DIR_DIR/ \
    --target-label CX1_class \
    --true-label above
