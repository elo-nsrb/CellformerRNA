import pandas as pd
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--savepath',
                        default="./data/",
                    help='Location to save pseudobulks data')
parser.add_argument('--filename',
                        default="./data/adata_peak_matrix.h5",
                    help='Anndata file')
parser.add_argument('--nb_cells_per_case',
                help="Number of synthetic samples created per individual",
                default=500, type=int)
parser.add_argument('--nb_cores',
                help="Number of cores",
                default=20, type=int)
parser.add_argument('--name', default="pseudobulks",
                    help='Name pseudobulks data')
parser.add_argument('--list_genes', default="./",
                    help='Name pseudobulks data')

#def main():

list_dataset = ["abi_mtg",
                #"abi_ctx",
                "rosmap2_f5",
                "berson",
                "mathys",
                #"rosemap",
                "agarwal",
                "franjic",
                "tran"]

suff_pseudo = "_pseudobulk_data.parquet.gzip"
suff_ct_spe = "_celltype_specific.npz"
suff_annotations = "_annotations.csv"
path_data = "/home/eloiseb/data/rna/adata_/"

df_pseudo = []
for it in list_dataset:
    pseudo = pd.read_parquet(os.path.join(path_data, it + suff_pseudo))
    df_pseudo.append(pseudo)

df_pseudo = pd.concat(df_pseudo)
list_samples = df_pseudo["Sample_num"].unique().tolist()
with open(os.path.join(path_data, "pseudobulks_list_samples.txt"), "w") as f:
    for it in list_samples:
        f.write("%s\n"%str(it))
df_pseudo.to_parquet(os.path.join(path_data, "pseudobulks" + suff_pseudo), index=None, compression="gzip")
df_pseudo = 0

df_pseudo = []
for it in list_dataset:
    print(it)
    pseudo = np.load(os.path.join(path_data, it + suff_ct_spe))["mat"]
    #if it =="agarwal":
    #    pseudo = np.insert(pseudo,5,0,axis=1)
    print(pseudo.shape)
    df_pseudo.append(pseudo)
pseudo = np.concatenate(df_pseudo,axis=0)
print(pseudo.shape)
np.savez_compressed(os.path.join(path_data, "pseudobulks" + suff_ct_spe),
                    mat=pseudo)##66000
pseudo=0

