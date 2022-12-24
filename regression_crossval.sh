for full_dir in /home/omarelnahhas/omars_hdd/omar/immunoproject/*
do
	dir=${full_dir##*/}
	echo "Running Fraction Genome Altered for ${dir}"
	python -m marugoto.mil crossval \
		--clini-excel /home/omarelnahhas/omars_hdd/omar/immunoproject_project/clini_positive_control/Fraction_GA.xlsx \
		--slide-csv ${full_dir}/*_SLIDE.csv  \
		--feature-dir ${full_dir}/xiyue-wang-macenko \
		--output-path /home/omarelnahhas/omars_hdd/omar/immunoproject_project/clini_positive_control/experiments/Fraction_GA/${dir} \
		--target-label Fraction_GA
done
