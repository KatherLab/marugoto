full_dir=/home/omarelnahhas/omars_hdd/HRD/CPTAC-FEATURES_test/CPTAC_UCEC_features
dir=${full_dir##*/}
target=LOH
for fold in 0_020r2 1_021r2 2_032r2 3_020r2 4_010r2
do
	python -m marugoto.mil deploy \
		--clini-table /home/omarelnahhas/HRD_project/CPTAC_CLINIC.xlsx \
		--slide-csv /home/omarelnahhas/omars_hdd/HRD/CPTAC-FEATURES_test/${dir}/*.csv \
		--feature-dir /home/omarelnahhas/omars_hdd/HRD/CPTAC-FEATURES_test/${dir}/xiyue-wang \
		--output-path /home/omarelnahhas/HRD_project/${target}/deploy_${dir}_DIR/fold-${fold} \
		--target-label "${target}" \
		--model-path /home/omarelnahhas/HRD_project/${target}/TCGA-UCEC-DX-features_DIR/fold-${fold}/*.pkl
done
