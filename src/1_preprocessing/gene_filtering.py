import pandas as pd
import os
import numpy as np
import anndata as ad
import argparse

parser = argparse.ArgumentParser(description='Filter low expressed genes')
parser.add_argument('-l', '--list_data', 
        help='list of paths to peak matrices (.h5ad), paths have to be separated by a comma',
        type=str)
parser.add_argument('-c', '--celltypes', 
                help='list of celltypes, items have to be separated by a comma',
                type=str)
parser.add_argument('-t', '--threshold', 
        help='list of celltypes',
        default=0.01,
        type=float)
parser.add_argument('-sp', '--savepath',
        help='Path to save the mask',
        default="./",
        type=str)



def createMask(list_dataset, celltypes, savepath, threshold = 0.01, key="celltype"):
    list_tot=[]
    list_sum=[]
    #celltypes = ["AST", "ENDO", "EXC", "INH", "MIC", "Mural", "OLD", "OPC"]
    for it in list_dataset:
        print(it)
        if it.endswith(".h5ad"):
            ad_ = ad.read_h5ad(it) 
            df_ = ad_.to_df()
            if "rosmap2" in it:
                df_.columns = ad_.var_names.values
            else:
                df_.columns = ad_.var["gene_symbol"].values
            df_[df_>0] = 1
            df_[key] = ad_.obs[key].values
            tot = df_.groupby([key]).size()
            sum_ = df_.groupby([key]).sum()
            list_tot.append(tot.reset_index())
            list_sum.append(sum_.reset_index())
        else:
            print("Path invalid, matrix filename has to end with .h5ad")
        
    df_sum =pd.concat(list_sum)
    df_tot =pd.concat(list_tot)

    df_sum = df_sum.groupby([key]).sum()
    df_tot = df_tot.groupby([key]).sum()
    mm = df_sum/df_tot.values
    mm = mm.loc[celltypes,:].astype("float")
    mm[mm.values<threshold] = 0
    mm[mm.values>=threshold] = 1
    mm.to_csv(os.path.join(savepath, "mask_thrs_%s.csv"%str(threshold)))
    print("Mask saved as %s"%os.path.join(savepath,
                "mask_thrs_%s.csv"%str(threshold)))
    return mm
def main():
    args = parser.parse_args()
    list_dataset = [item for item in args.list_data.split(',')]
    celltypes = [item for item in args.celltypes.split(',')]
    #list_dataset = ["/home/eloiseb/data/rna/rna_pbmc/adata_peak_matrix_ding_pbmc.h5ad"]
    #celltypes = ["B cell","CD4+ T cell","CD14+ monocyte","CD16+ monocyte", "Cytotoxic T cell", "Dendritic cell", "Megakaryocyte", "Natural killer cell", "Plasmacytoid dendritic cell"]
    savepath = args.savepath
    #savepath = "/home/eloiseb/data/rna/rna_pbmc/"
    thres = args.threshold
    key = "celltype_map1"
    createMask(list_dataset, celltypes, savepath, threshold = 0.01, key=key)

if __name__ == "__main__":
    main()
