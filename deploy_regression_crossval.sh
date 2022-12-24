full_dir=/home/omarelnahhas/omars_hdd/omar/immunoproject_project/top5_regression_binarybalanced/*
for dir in /home/omarelnahhas/omars_hdd/omar/immunoproject_project/top5_regression_binarybalanced/*
do
target=${dirs##*/}
for fold in 0 1 2 3 4
do
	python -m marugoto.mil deploy \
		--clini-table /home/omarelnahhas/omars_hdd/omar/FOCUS/FOCUS1_CLINI.xlsx \
		--slide-csv /home/omarelnahhas/omars_hdd/omar/FOCUS/*_SLIDE.csv \
		--feature-dir /home/omarelnahhas/omars_hdd/omar/FOCUS/Xiyue-Wang_v2 \
		--output-path ${dir}/deploy_${target}_FOCUS_CRC/fold-${fold} \
		--target-label "dummy" \
		--model-path "${dir}"/TCGA-CRC-DX-features/fold-${fold}/*.pkl
done
done
