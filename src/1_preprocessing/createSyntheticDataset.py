import os
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import multiprocessing
from joblib import Parallel, delayed
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
parser.add_argument('--list_genes', default=None,
                    help='Name pseudobulks data')

def createCellTypeFractionType(nb_celltypes):
    """
    Draw a random fraction of cells
    """
    fracs = np.random.rand(nb_celltypes)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs

def sampleDataset(data, labels, nb_to_samples,
                sample_size, celltypes, with_sparse=False,
                max_removed_celltypes=None,
                only_sparse=False, pure=False,
                nb_sparse=None,
                random_num_cell=False):
    """
    Create synthetic cell type specific mixture
    """

    if sample_size is None:
        random_num_cell = True
    if max_removed_celltypes is None:
        max_removed_celltypes = len(celltypes)
    sim_y = []
    sim_x = []
    sample_size_list = []
    if not only_sparse:
        for i in range(nb_to_samples):
            if random_num_cell:
                sample_size = np.random.randint(100, 800)
                #sample_size = np.random.randint(30, 200)
            artificial_sample, df_mix, lab = createBulkSample(data, labels,
                                                                sample_size,
                                                                celltypes)
            if i ==0:
                separate_sample = artificial_sample[np.newaxis,:,:]
            else:
                separate_sample = np.concatenate([separate_sample,
                                        artificial_sample[np.newaxis,:,:]],
                                        axis=0)

            sim_x.append(df_mix)
            sim_y.append(lab)
            sample_size_list.append(sample_size)

    # Create sparse samples
    if with_sparse or only_sparse:
        for i in range(nb_sparse): #nb_to_samples):
            if random_num_cell:
                sample_size = np.random.randint(100, 800)
                #sample_size = np.random.randint(30, 200)
            artificial_sample, df_mix, lab = createBulkSample(data,
                    labels, sample_size,
                    celltypes, sparse=True,
                    pure=pure,
                    max_removed_celltypes=max_removed_celltypes)
            if i ==0 and only_sparse:
                separate_sample = artificial_sample[np.newaxis,:,:]
            else:
                separate_sample = np.concatenate([separate_sample,
                    artificial_sample[np.newaxis,:,:]], axis=0)
            sim_x.append(df_mix)
            sim_y.append(lab)
            sample_size_list.append(sample_size)

    sim_x = pd.concat(sim_x, axis=1).T
    sim_y = pd.DataFrame(sim_y, columns=celltypes)
    return sim_x, sim_y, separate_sample, sample_size_list



def createBulkSample(data, label,sample_size, celltypes, sparse=False,
                max_removed_celltypes=None, only_sparse=False, pure=False):

    """
    Create a synthetic subject specific mixture
    """
    if max_removed_celltypes is None:
        max_removed_celltypes = len(celltypes)
    available_celltypes = celltypes#.tolist()
    if sparse:
        no_keep = np.random.randint(1, max_removed_celltypes)
        if pure:
            no_keep=1
        keep = np.random.choice(list(range(len(available_celltypes))),
                                size=no_keep,
                                replace=False)
        available_celltypes = [available_celltypes[i] for i in keep]
    check_availability = []
    for i, ct in enumerate(available_celltypes):
        cells_sub = data[np.array(label== ct), :]
        if cells_sub.shape[0]>0:
            check_availability.append(i)

    #nb_available_cts = len(available_celltypes)
    nb_available_cts = len(check_availability)

    fracs_red = createCellTypeFractionType(nb_available_cts)
    samp_fracs_red = np.multiply(fracs_red, sample_size)
    samp_fracs_red =list(map(int, samp_fracs_red))

    samp_fracs = [0]*len(celltypes)
    fracs = [0]*len(celltypes)
    for idx, sf in enumerate(samp_fracs_red):
        samp_fracs[check_availability[idx]] = sf
        fracs[check_availability[idx]] = fracs_red[idx]


    fracs_complete = [0]*len(celltypes)
    for i, act in enumerate(available_celltypes):
        fracs_complete[i] = fracs[i]

    artificial_samples = np.zeros((len(celltypes), data.shape[1]))
    for i in range(nb_available_cts):
        ct = available_celltypes[i]
        cells_sub = data[np.array(label== ct), :]
        cells_fraction = np.random.randint(0, cells_sub.shape[0],
                                             samp_fracs[i])
        cells_sub = cells_sub[cells_fraction, :]
        artificial_samples[i] = cells_sub.sum(axis=0)#pure mix

    #df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = pd.DataFrame(artificial_samples, index=celltypes)
    df_mix = df_samp.sum(axis=0)

    return artificial_samples, df_mix, fracs_complete


def loopPerSample(it,i, dataset, subjects, nb_genes, dir_path,
                        celltypes, key_celltype="celltype",
                        nb_sample_per_subject=1000,sample_size=None,
                        with_sparse=True, only_sparse=False, savename="",
                        nb_sparse=None,
                        pure=False):
    if not os.path.exists(dir_path
                            + savename
                            + "_celltype_specific_subject_%s.npz"%it):
        data = dataset[dataset.obs["Sample_num"]==it]
        print("subject num %d / %d"%(i, len(subjects)))
        lab = data.obs[key_celltype]
        (df_mix, df_labels,
         separate_signals,
         sample_size_list) = sampleDataset(data.X,
                                    lab, nb_sample_per_subject,
                                    sample_size,
                                    celltypes,
                                    with_sparse=with_sparse,
                                    pure=pure,
                                    nb_sparse=nb_sparse,
                                    only_sparse=only_sparse)
        df_mix["Sample_num"] = it
        print(separate_signals.shape)
        if only_sparse:
            if pure:
                savename += "__pure_pure_"
            else:
                savename += "_only_sparse_"
        if not with_sparse:
                savename += "_100-800_"
        df_mix.to_csv(dir_path
                        + savename
                        + "_pseudobulk_data_subject_%s.csv"%it,
                        index=False)
        df_labels.to_csv(dir_path
                        + savename
                        + "_labels_data_subject_%s.csv"%it,
                        index=False)
        np.savez_compressed(dir_path
                            + savename
                            + "_celltype_specific_subject_%s.npz"%it,
                            mat=separate_signals)
        np.save(dir_path + savename + "_nb_cells_per_mixtures_%s.npy"%it,
                np.asarray(sample_size_list))


def createALLBulkDataset(dataset, nb_genes, dir_path,
                        celltypes,key_celltype,
                        save_per_samples=True,
                        nb_sample_per_subject=10000,
                        sample_size=None,
                        with_sparse=True,
                        nb_sparse=500,
                        only_sparse=False, savename="",
                        pure=False, num_cores=3):

    print(nb_sparse)

    subjects = dataset.obs["Sample_num"].unique()
    if save_per_samples:
        Parallel(n_jobs=num_cores)(delayed(loopPerSample)(it, i,
                        dataset, subjects, nb_genes, dir_path,
                        celltypes, key_celltype,
                        nb_sample_per_subject=nb_sample_per_subject,
                        sample_size=sample_size,
                        with_sparse=with_sparse,
                        nb_sparse=nb_sparse,
                        only_sparse=only_sparse, savename=savename,
                        pure=pure) for i,it in enumerate(subjects))
        return None, None, None

    else:
        for i,it in enumerate(subjects):
            data = dataset[dataset.obs["Sample_num"]==it]
            print("subject num %d / %d"%(i, len(subjects)))
            lab = data.obs[key_celltype].values
            if i==0:
                (df_mix, df_labels,
                 separate_signals,
                 sample_size_list) = sampleDataset(data.X,
                                            lab, nb_sample_per_subject,
                                            sample_size,
                                            celltypes,
                                            with_sparse=with_sparse,
                                            nb_sparse=nb_sparse,
                                            pure=pure,
                                            only_sparse=only_sparse)
                df_mix["Sample_num"] = it
            else:
                (df_tmp,
                 labels_tmp,
                 separate_signals_tmp,
                 sample_size_list_tmp) = sampleDataset(
                                            data.X, lab,
                                            nb_sample_per_subject,
                                            sample_size,
                                            celltypes,
                                            with_sparse=with_sparse,
                                            pure=pure,
                                            nb_sparse=nb_sparse,
                                            only_sparse=only_sparse)
                df_tmp["Sample_num"] = it
                df_mix = pd.concat([df_mix, df_tmp], axis=0)
                df_labels = pd.concat([df_labels, labels_tmp], axis=0)
                separate_signals = np.concatenate([separate_signals, separate_signals_tmp], axis=0)
                sample_size_list += sample_size_list_tmp
        print(separate_signals.shape)
        if only_sparse:
            if pure:
                savename += "__pure_pure_"
            else:
                savename += "_only_sparse_"
        if not with_sparse:
                savename += "_100-800_"
        df_mix.index= df_mix.index.astype(str)
        df_mix.columns= df_mix.columns.astype(str)
        df_mix.to_parquet(os.path.join(dir_path, savename + "_pseudobulk_data.parquet.gzip"),
                        compression="gzip", index=False)
        df_mix[["Sample_num"]].to_csv(os.path.join(dir_path, savename + "_list_samples.csv"),
                                    header=False, index=False)
        df_labels.to_csv(os.path.join(dir_path , savename + "_labels_pseudobulk_data.csv"), index=False)
        np.savez_compressed(os.path.join(dir_path , savename + "_celltype_specific.npz"), mat=separate_signals)
        np.save(os.path.join(dir_path , savename + "_nb_cells_per_mixtures.npy"),
                np.asarray(sample_size_list))
        return df_mix, df_labels, separate_signals

def main():
    args = parser.parse_args()
    #allow_celltype = ["AST", "ENDO", "EXC", "INH", "MIC", "Mural", "OLD", "OPC"]
    savepath = args.savepath
    filename = args.filename
    name = args.name
    nb_cell_per_case = int(args.nb_cells_per_case)
    nb_cores = int(args.nb_cores)
    print(nb_cores)
    adata_ctrl = ad.read_h5ad(filename)
    if "rosmap2" in name:
        adata_ctrl.var["gene_symbol"] = adata_ctrl.var_names.values
        nb_sparse=2
        key = "celltype_map1"
    elif "pbmc" in name:
        nb_sparse=int(nb_cell_per_case/5)
        key = "celltype"
        celltype = ['B cell', 'CD4+ T cell', 'CD14+ monocyte', 'CD16+ monocyte',
                       'Cytotoxic T cell', 'Dendritic cell', 'Megakaryocyte']
        adata_ctrl = adata_ctrl[adata_ctrl.obs["celltype"].isin(celltype)]
    else:
        nb_sparse=int(nb_cell_per_case/10)
        key = "celltype_map1"
    #adata_ctrl = adata_ctrl[adata_ctrl.obs.celltype.isin(allow_celltype)]
    if args.list_genes is not None:
        list_genes = pd.read_csv(args.list_genes)["Gene"].tolist()
        adata_ctrl = adata_ctrl[:, adata_ctrl.var["gene_symbol"].isin(list_genes)]
    adata_ = adata_ctrl
    #adata_.obs.drop("celltype", axis=1, inplace=True)
    if "totnorm_lognorm" in name:
        print("total norm+log norm data")
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)
    elif ("lognorm" in name) & (not "totnorm_lognorm" in name):
        print("log norm data")
        sc.pp.log1p(adata_)
    print(adata_)

    annot = adata_.var
    annot.to_csv(os.path.join(savepath , name + "_annotations.csv"))
    adata_ = adata_[~adata_.obs[key].isna()]
    celltypes = adata_.obs[key].sort_values().unique().tolist()
    #celltypes = ["AST", "ENDO", "EXC", "INH", "MIC", "Mural", "OLD", "OPC"]
    print(celltypes)
    print(nb_cell_per_case)
    print(nb_sparse)
    mean_exp = adata_.X.mean(0)
    np.save(os.path.join(savepath, name + "_mean_exp.npy"), mean_exp)
    df, df_lables, separate_signals = createALLBulkDataset(
                            adata_,
                            adata_.X.shape[1], savepath,
                            celltypes, key_celltype=key,
                            save_per_samples=False,
                            nb_sample_per_subject=nb_cell_per_case,
                            sample_size=None,
                            savename=name,
                            with_sparse=True,
                            nb_sparse=nb_sparse,
                            num_cores=nb_cores
                            )
    #df, df_lables, separate_signals = createALLBulkDataset(adata_,
    #                        adata_.X.shape[1], dir_path,
    #                        celltypes,save_per_samples=True,
    #                        nb_sample_per_subject=100,sample_size=None,
    #                        savename=name, only_sparse=True, pure=True)


if __name__ == "__main__":
    main()








