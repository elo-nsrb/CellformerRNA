
#python createSyntheticDataset.py --savepath ~/data/rna/rna_pbmc/ --filename ~/data/rna/rna_pbmc/adata_peak_matrix_ding_pbmc.h5ad --nb_cells_per_case 1000 --nb_cores 30 --name pbmc_7 
#python createSyntheticDataset.py --savepath ~/data/rna/rna_pbmc/ --filename ~/data/rna/rna_pbmc/adata_peak_matrix_ding_pbmc.h5ad --nb_cells_per_case 1000 --nb_cores 30 --name pbmc_7_lognorm 
#python createSyntheticDataset.py --savepath ~/data/rna/rna_pbmc/ --filename ~/data/rna/rna_pbmc/adata_peak_matrix_ding_pbmc.h5ad --nb_cells_per_case 1000 --nb_cores 30 --name pbmc_7_totnorm_lognorm 

#python createSyntheticDataset.py --savepath ~/data/rna/adata_/ --filename /home/eloiseb/home_nalab6/data/rna/adata_/adata_gene_matrix_Berson_compressed_filtered.h5ad --nb_cells_per_case 1000 --nb_cores 30 --name berson_own_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_Berson_compressed_filtered.h5ad --nb_cells_per_case 1000 --nb_cores 30 --name berson_own_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath ~/data/rna/AD_rna_deconvolution --filename ~/data/rna/adata_/adata_gene_matrix_Berson_compressed_filtered.h5ad --nb_cells_per_case 2000 --nb_cores 20 --name our_only_all_lognorm
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename ~/data/rna/adata_/adata_gene_matrix_Berson_compressed_filtered.h5ad --nb_cells_per_case 2000 --nb_cores 20 --name our_only_all_totnorm_lognorm


#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_tran_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name tran_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_tran_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name tran_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_Berson_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name berson_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_Berson_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name berson_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_mathys_compressed_filtered.h5ad --nb_cells_per_case 250 --nb_cores 30 --name mathys_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_mathys_compressed_filtered.h5ad --nb_cells_per_case 250 --nb_cores 30 --name mathys_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_franjic_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name franjic_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_franjic_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name franjic_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /remote/test/eloiseb/data/rna/abi_mtg_tmp_ctl_only.h5ad --nb_cells_per_case 100 --nb_cores 30 --name abi_mtg_ctl_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /remote/test/eloiseb/data/rna/abi_mtg_tmp_ctl_only.h5ad --nb_cells_per_case 100 --nb_cores 30 --name abi_mtg_ctl_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /remote/test/eloiseb/data/rna/abi_mtg_tmp_ad_only.h5ad --nb_cells_per_case 100 --nb_cores 30 --name abi_mtg_ad_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /remote/test/eloiseb/data/rna/abi_mtg_tmp_ad_only.h5ad --nb_cells_per_case 100 --nb_cores 30 --name abi_mtg_ad_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_agarwal_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name agarwal_7_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_agarwal_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name agarwal_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath ~/data/rna/adata_/ --filename ~/data/rna/adata_/adata_gene_matrix_abi_ctx_compressed_filtered.h5ad --nb_cells_per_case 500 --nb_cores 30 --name abi_ctx --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv

#python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_rosmap2_filt5_compressed_filtered.h5ad --nb_cells_per_case 20 --nb_cores 30 --name rosmap2_f5_lognorm7 --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv
python createSyntheticDataset.py --savepath /remote/test/eloiseb/data/rna/adata_/ --filename /home/eloiseb/data/rna/adata_/adata_gene_matrix_rosmap2_filt5_compressed_filtered.h5ad --nb_cells_per_case 20 --nb_cores 30 --name rosmap2_f5_7_totnorm_lognorm --list_genes ~/data/rna/common_genes_new_all_wo_new_data.csv


