full_dir=/home/omarelnahhas/omars_hdd/HRD/TCGA-FEATURES_train/TCGA-UCEC-DX-features
dir=${full_dir##*/}
for target in CX8 CX9 CX10 CX11 CX12 CX13 CX14 CX15 CX16 CX17
do
	python -m marugoto.mil crossval \
		--clini-excel /home/omarelnahhas/TCGA_Markowetz/norm_CLINI_Markowetz.csv \
		--slide-csv /home/omarelnahhas/TCGA_Markowetz/tcga_merkowetz_SLIDE.csv  \
		--feature-dir /mnt/Sirius_03_empty/TCGA-ALL-FEATURES/ \
		--output-path /home/omarelnahhas/TCGA_Markowetz/${target}/ALL_FEATS_DIR \
		--target-label "${target}"
done
