full_dir=/home/omarelnahhas/omars_hdd/HRD/CPTAC-FEATURES_test/CPTAC_UCEC_features
dir=${full_dir##*/}
target=HRD_sum
python -m marugoto.mil deploy \
		--clini-table /home/omarelnahhas/HRD_project/CPTAC_CLINIC.xlsx \
		--slide-csv /home/omarelnahhas/omars_hdd/HRD/CPTAC-FEATURES_test/${dir}/*.csv \
		--feature-dir /home/omarelnahhas/omars_hdd/HRD/CPTAC-FEATURES_test/${dir}/xiyue-wang \
		--output-path /home/omarelnahhas/HRD_project/${target}/deploy_${dir}_DIR_trainfull/ \
		--target-label "${target}" \
		--model-path /home/omarelnahhas/HRD_project/HRD_sum/TCGA-UCEC-DX-features_DIR/*.pkl
