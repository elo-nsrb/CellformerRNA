import pandas as pd
import os
import numpy as np

list_dataset = ["abi_mtg", 
                "abi_ctx",
                "mathys",
                "rosemap",
                "agarwal",
                "franjic",
                "tran"]

pref_pseudo = "pseudobulkSNRNA_"
pref_ct_spe = "cellspecificSNRNA_"
path_data = "/home/eloiseb/data/rna/pseudobulks_sum/" 


df_pseudo = []
df_mix = []
celltypes = ["AST", "ENDO", "EXC", "INH", "MIC", "Mural", "OLD", "OPC"]
for it in list_dataset:
    print(it)
    pseudo = pd.read_csv(os.path.join(path_data, pref_ct_spe + it + "_celltype_sum.csv"))
    pseudo = pseudo[pseudo.celltype.isin(celltypes)]
    pseudo["dataset"] = it
    print(pseudo["celltype"].unique())
    df_pseudo.append(pseudo)
    tmp = pseudo.drop("celltype", axis=1)
    tmp = tmp.groupby(["Sample_num"], group_keys=False).sum()
    df_mix.append(tmp)

df_mix = pd.concat(df_mix)
df_mix.reset_index().to_csv(os.path.join(path_data, pref_pseudo + "ok.csv"), index=None)
df_mix.T.to_csv(os.path.join(path_data, pref_pseudo + ".csv"), index=None)
df_mix = df_mix.reset_index()
list_samples = df_mix["Sample_num"].unique().tolist()
with open(os.path.join(path_data, "pseudobulks_list_samples.txt"), "w") as f:
    for it in list_samples:
        f.write("%s\n"%str(it))
pseudo = pd.concat(df_pseudo)
cell_info = pseudo[["Sample_num", "celltype"]]
cell_info.columns = ["cell_id","cell_type"]
cell_info["cell_subtype"] = cell_info["cell_type"].values
cell_info["tumor_flag"] = 0
cell_info["dataset"] = pseudo["dataset"].values
pseudo.drop("dataset", axis=1, inplace=True)
pseudo.to_csv(os.path.join(path_data, pref_ct_spe + "ok.csv"), index=None)
X = np.zeros((pseudo["Sample_num"].nunique(),
            pseudo["celltype"].nunique(),
            df_mix.drop("Sample_num", axis=1).shape[-1]))
celltypes = pseudo["celltype"].sort_values().unique()
print(celltypes)
for i, sp in enumerate(list_samples):
    for j, ct in enumerate(celltypes):
        tmp = pseudo[(pseudo.celltype==ct)&(pseudo["Sample_num"]==sp)].drop(
                        ["Sample_num", "celltype"], axis=1).values
        if len(tmp) == 0:
            print(ct)
            tmp = np.zeros(X[i,j,:].shape)
        X[i,j,:] = tmp

np.savez_compressed(os.path.join(path_data, pref_ct_spe + "ok.npz"),
                    mat=X)

pseudo.drop("celltype", axis=1, inplace=True) 
pseudo.set_index("Sample_num").T.to_csv(os.path.join(path_data, pref_ct_spe + ".csv"), index=None)
cell_info.to_csv(os.path.join(path_data,"cellinfo.csv"), index=None)
pseudo=0

