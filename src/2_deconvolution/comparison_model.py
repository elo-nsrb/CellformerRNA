import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
#from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from visu_func import *

#import gc
#import cudf
#from cuml.ensemble import RandomForestRegressor
#from cuml.linear_model import LinearRegression
#from cuml.neighbors import KNeighborsRegressor
#from dask.distributed import Client
#from dask_cuda import LocalCUDACluster
#cluster = LocalCUDACluster()

def disabling_blas():
    n_threads = 1
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = ""
    # tf.config.threading.set_inter_op_parallelism_threads(1)

disabling_blas() # to run purely on cpu
#from keras.models import Sequential
#from keras.layers import Dense
from joblib import Parallel, delayed

def nmf_decomposition(X, X_gt):

    start = time.time()
    model = NMF(n_components=X_gt.shape[1],
                init='random',
                random_state=0)
    W = model.fit_transform(X)#n_samples, n_component
    H = model.components_#n_component, n_features
    pred = W[:,:,np.newaxis]*H[np.newaxis,:,:]
    end = time.time() - start
    return pred, end

def get_model(n_inputs, n_outputs):
    # set reproducibility
    #import tensorflow as tf
    #from tensorflow.keras.callbacks import EarlyStopping
    #from tensorflow.keras.initializers import glorot_normal

    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs,
            kernel_initializer='he_uniform', activation='relu'))
    #model.add(Dense(64, input_dim=512,
    #        kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_model( X_train, y_train, X_test, y_test, clf_name):
        clf = get_clf(clf_name,
                        n_inputs=X_train.shape[1],
                        n_outputs=y_train.shape[1])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #del clf; _=gc.collect()
        return y_pred, y_test

def cv_iteration(x_, y_, i, train_index, test_index, clf_name, celltypes, groups):
        start = time.time()
        X_train, X_test = x_[train_index, :], x_[test_index,:]
        y_train, y_test = y_[train_index, :, :], y_[test_index, :, :]
        nb_ = y_.shape[1]
        out = Parallel(n_jobs=5, verbose=1)(
                delayed(train_model)(
                X_train, 
                y_train[:,i,:], X_test,
                y_test[:,i,:],
                clf_name) for i in range(nb_))
        separate_pred = []
        label_ = groups[test_index]
        separates_pred = np.zeros_like(y_test)
        for j,it in enumerate(out):
            separates_pred[:,j,:] = it[0]

        end = time.time() - start
        #separates = np.concatenate(separate_)
        #separates_pred = np.concatenate(separate_pred)
        df_metrics_per_subject= compute_metrics_per_subject(separates_pred,
                            y_test,
                            celltypes,
                            label_)
        df_metrics_per_subject["fold"] = "fold_%s"%str(i)
        df_metrics_per_subject["time"] =end
        df_metrics_per_it= compute_metrics(separates_pred,
                            y_test,
                            celltypes)
        df_metrics_per_it["fold"] = "fold_%s"%str(i)
        df_metrics_per_it["time"] =end
        df_metrics_per_genes= compute_metrics_per_genes(separates_pred,
                            y_test,
                            celltypes,
                            list(np.arange(separates_pred.shape[-1])))
        df_metrics_per_genes["fold"] = "fold_%s"%str(i)
        df_metrics_per_genes["time"] =end
        return df_metrics_per_subject, df_metrics_per_it, df_metrics_per_genes


def multiOut_regression(X, X_gt, groups,
                        clf_name,
                        celltypes,
                        cv_func="LOGO",
                        K=10):

    X_pred = np.zeros_like(X_gt)
    list_times=[]
    df_metrics_sub_list = []
    df_metrics_it_list = []
    df_metrics_genes_list = []
    #for i in tqdm(range(X_gt.shape[1])):
    x_ = X
    y_ = X_gt#[:,i,:]
    if True:
        #label = celltypes[i]
        if cv_func == "LOO":
            loo = LeaveOneOut()
            list_splits = loo.split(x_)
        elif cv_func == "KStratFold":
            kfold =  StratifiedKFold(n_splits = K,
                                    shuffle=True,
                                    random_state=23)
            list_splits = kfold.split(x_, y_)
        elif cv_func == "KStratGroupFold":
            kfold =  StratifiedGroupKFold(n_splits = K,
                                    shuffle=True,
                                    random_state=23)
            list_splits = kfold.split(x_, y_, groups)
        elif cv_func == "KFold":
            kfold =  KFold(n_splits = K,
                                shuffle=True,
                                random_state=23)
            list_splits = kfold.split(x_, y_)
        elif cv_func == "LOGO":
            logo =  LeaveOneGroupOut()
            list_splits = logo.split(x_, y_, groups)
        else:
            NotImplementedError
        #out = []
        #for train_index, test_index in list_splits:
        #    out.append(cv_iteration(
        #        x_,
        #        y_, train_index,
        #        test_index,
        #        clf_name))
        out = Parallel(n_jobs=10, verbose=1)(
                delayed(cv_iteration)(
                X,
                X_gt, i, train_index,
                test_index,
                clf_name, celltypes, groups) for i, (train_index, test_index) in enumerate(list_splits))
        print("training done")
        for j,it in enumerate(out):
            df_metrics_sub_list.append(it[0])
            df_metrics_it_list.append(it[1])
            df_metrics_genes_list.append(it[2])

            #X_pred[it[1],i,:] = it[0]
    df_metrics_it = pd.concat(df_metrics_it_list)
    df_metrics_sub = pd.concat(df_metrics_sub_list)
    df_metrics_genes = pd.concat(df_metrics_genes_list)
    return df_metrics_it, df_metrics_sub, df_metrics_genes

def multiOut_regression_Comp(X_train, X_gt_train,
                                groups,
                             X_test, X_gt_test,
                             groups_test,
                                clf_name,
                                K=5,
                        cv_func="LOGO"):

    X_pred = np.zeros_like(X_gt_test)
    list_times=[]
    for i in range(X_gt_train.shape[1]):
        x_ = X_train
        y_ = X_gt_train[:,i,:]
        if cv_func == "LOO":
            loo = LeaveOneOut()
            list_splits = loo.split(x_)
        elif cv_func == "KStratFold":
            kfold =  StratifiedKFold(n_splits = K,
                                shuffle=True,
                                random_state=23)
            list_splits = kfold.split(x_, y_)
        elif cv_func == "KFold":
            kfold =  KFold(n_splits = K,
                                shuffle=True,
                                random_state=23)
            list_splits = kfold.split(x_, y_)
        elif cv_func == "KStratGroupFold":
            kfold =  StratifiedGroupKFold(n_splits = K,
                                shuffle=True, random_state=23)
            list_splits = kfold.split(x_, y_, groups)
        elif cv_func == "LOGO":
            logo =  LeaveOneGroupOut()
            list_splits = logo.split(x_, y_, groups)
        else:
            NotImplementedError
        for train_index,test_index in list_splits:
            try:
                start = time.time()
                x_train, x_test = x_[train_index], x_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
                print(test_index)
                test_group = np.asarray(groups)[test_index]
                test_index_i = [i for i, e in enumerate(groups_test) if e in test_group]
                print(test_group)
                clf = get_clf(clf_name,
                                n_inputs=x_train.shape[1],
                                n_outputs=y_train.shape[1])
                clf.fit(x_train, y_train)
                X_pred[test_index_i,i,:] = clf.predict(X_test[test_index_i])
                end = time.time() - start
                print(end)
                list_times.append(end)
            except:
                __import__('ipdb').set_trace()
    mean_time = np.mean(list_times)
    return X_pred, mean_time
def multiOut_regressionHOldOut(X, X_gt, train_index,
                            test_index, clf_name):

    X_pred = np.zeros_like(X_gt)
    for i in range(X_gt.shape[1]):
        x_ = X
        y_ = X_gt[:,i,:]
        X_train, X_test = x_[train_index], x_[test_index]
        y_train, y_test = y_[train_index], y_[test_index]
        clf = get_clf(clf_name,
                            n_inputs=X_train.shape[1],
                            n_outputs=y_train.shape[1])
        clf.fit(X_train, y_train)
        X_pred[test_index,i,:] = clf.predict(X_test)
        X_pred[train_index,i,:] = clf.predict(X_train)
    return X_pred

def get_clf(clf_name, n_inputs=None, n_outputs=None, n_jobs=1):
    if clf_name=="LinearRegression":
        return LinearRegression()#n_jobs=n_jobs)
    elif clf_name=="knn":
        return KNeighborsRegressor()#n_jobs=n_jobs)
    elif clf_name=="RandomForestRegressor":
        return RandomForestRegressor(max_depth=4,
                                    random_state=23, n_jobs=n_jobs)
        #return MultiOutputRegressor(RandomForestRegressor(max_depth=4,
        #                            n_streams=1,
        #                           bootstrap=False,
        #                           n_streams=1,
        #                            random_state=2))
    elif clf_name=="DecisionTree":
        return DecisionTreeRegressor( random_state=23)
    elif clf_name=="XGBoost":
        return MultiOutputRegressor(XGBRegressor(
                    objective = 'reg:squarederror'
                    ), n_jobs=n_jobs)
    elif clf_name=="SVM":
        model = LinearSVR()
        # define the direct multioutput wrapper model
        return MultiOutputRegressor(model)
    elif clf_name=="MLP":
        print("MLP")
        return get_model(n_inputs,n_outputs)



def compute_metrics(X_pred, X_gt, celltypes,
        metrics=["spearman", "pearson","mse"]): #, "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics", "res"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            res  = get_metrics(X_gt[:,i,:], X_pred[:,i,:], met)
            df_metrics.loc[len(df_metrics),:] = [ct, met, res]
    return df_metrics

def compute_metrics_per_subject(X_pred, X_gt, celltypes,list_sub,
            metrics=["spearman","pearson", "mse"]):# "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics",
                                    "res", "individualID"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            for j,sb in enumerate(list_sub):
                res  = get_metrics(X_gt[j,i,:], X_pred[j,i,:], met)
                df_metrics.loc[len(df_metrics),:] = [ct, met, res, sb]
    return df_metrics

def get_metrics_par(X, X_pred, met, ct, genes):
    res  = get_metrics(X, X_pred, met)
    return res, ct, genes
def compute_metrics_per_genes(X_pred, X_gt, celltypes,list_genes,
        metrics=["spearman","pearson", "mse"]): #, "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics",
                                    "res", "genes"])
    for met in metrics:
        out = Parallel(n_jobs=5, verbose=1)(
                delayed(get_metrics_par)(
                        X_gt[:,i,j],
                        X_pred[:,i,j],
                        met, ct, sb) for i,ct in enumerate(celltypes) for j,sb in enumerate(list_genes))
        for kl,it in enumerate(out):
            df_metrics.loc[len(df_metrics),:] = [it[1], met, it[0], it[2]]
    return df_metrics
def get_metrics(X_gt, X_pred, metric):
    if metric=="spearman":
        if len(X_gt.shape)>1:
            ress = []
            for it in range(X_gt.shape[0]):
                res, _ = spearmanr(X_gt[it, :], X_pred[it, :], axis=None)
                if np.isnan(res):
                    res = 0
                ress.append(res)
            res= np.mean(ress)
        else:
            res, _ = spearmanr(X_gt, X_pred)


    elif metric=="mse":
        res = mean_squared_error(X_gt, X_pred)
    elif metric=="pearson":
        if len(X_gt.shape)>1:
            ress = []
            for it in range(X_gt.shape[0]):
                res, _ = pearsonr(X_gt[it,:], X_pred[it,:])
                if np.isnan(res):
                    res = 0
                ress.append(res)
            res= np.mean(ress)
        else:
            res, _ = pearsonr(X_gt, X_pred)
    elif metric=="R2":
        res = r2_score(X_gt, X_pred)
    elif metric=="auc":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        res = metrics.roc_auc_score(b_gt.flatten(),
                                    X_pred.flatten())
    elif metric=="auprc":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        res = metrics.average_precision_score(b_gt.flatten(),
                                    X_pred.flatten())
    return res

def get_metrics_per_subject(X_gt, X_pred, metric):
    if metric=="spearman":
        ress = []
        for it in range(X_gt.shape[0]):
            res, _ = spearmanr(X_gt[it,:], X_pred[it,:], axis=None)
            #if np.isnan(res):
            #    res = 0
            ress.append(res)

    elif metric=="mse":
        ress = []
        for it in range(X_gt.shape[0]):
            res = mean_squared_error(X_gt[it,:], X_pred[it,:])
            ress.append(res)
    elif metric=="pearson":
        ress = []
        for it in range(X_gt.shape[0]):
            res, _ = pearsonr(X_gt[it,:], X_pred[it,:])
            ress.append(res)
    elif metric=="R2":
        ress = []
        for it in range(X_gt.shape[0]):
            res = r2_score(X_gt[it,:], X_pred[it,:])
            ress.append(res)
    elif metric=="auc":
        ress = []
        for it in range(X_gt.shape[0]):
            b_gt = np.zeros_like(X_gt[it,:])
            b_gt[X_gt[it,:]>0] = 1
            res = metrics.roc_auc_score(b_gt.flatten(),
                                        X_pred[it,:].flatten())
            ress.append(res)
    elif metric=="auprc":
        ress = []
        for it in range(X_gt.shape[0]):
            b_gt = np.zeros_like(X_gt[it,:])
            b_gt[X_gt[it,:]>0] = 1
            res = metrics.average_precision_score(b_gt.flatten(),
                                        X_pred[it,:].flatten())
            ress.append(res)
    return res


def main():
    scg = False#True
    sherlock = False
    if scg:
        pp_prefix = "/labs/tmontine/eloiseb/"
    elif sherlock:
        pp_prefix = "/oak/stanford/scg/lab_tmontine/eloiseb/"
    else:
        pp_prefix = "/remote/home/eloiseb/data/"
    path_save= os.path.join(pp_prefix, "rna/save_runs/")
    print(path_save)
    key =""
    ctrl_only = False
    with_real_bulk = False
    with_synthetic = True
    logo = False
    with_filter = False
    normalize=True
    groupby = "brain_region"
    sampleid = "Sample_num"
    savename = path_save + "ML_univeral_20_%s"%key
    if with_filter:
        savename += "_with_filter_"
    if ctrl_only:
        savename = path_save + "CTRL_brain_%s"%key
    elif with_real_bulk:
        savename = path_save + "Test_real_bulk_brain_%s"%key
    elif with_synthetic:
        nb_samples=20
        savename += "_with_synthetic_%s_"%str(nb_samples)
    if normalize:
        savename += "_normalize_"
    if logo:
        savename += "_logo_%s"%groupby

    print(savename)
    if not os.path.exists(savename + ".csv"):
        if with_synthetic:

            X = np.load(os.path.join(pp_prefix,
                #"rna/AD_rna_deconvolution/berson_map1_18k_totnorm_lognorm_nosparse_1inh_100-800__celltype_specific.npz"))["mat"]
                "rna/adata_/pseudobulks_nosparse_celltype_specific.npz"))["mat"]
            inp = pd.read_parquet(os.path.join(pp_prefix,
                "rna/adata_/pseudobulks_nosparse_pseudobulk_data.parquet.gzip"))

            inp = inp.groupby("Sample_num").sample(nb_samples,
                                        replace=True,
                                        random_state=23)
            print(inp.shape)
            X = X[inp.index.tolist(), :, :]
            genes = inp.drop("Sample_num", axis=1).columns.tolist()
            celltypes = ["AST", "Endo-Mural", "EXC-L23","EXC-L4",
                    "EXC-L5","EXC-L6", "INH-CGE","INH-MGE", "MIC", "OLD",
                        "OPC"]
            celltypes = ["AST", "ENDO-Mural", "EXC",
                     "INH", "MIC", "OLD",
                        "OPC"]
            features = inp.drop("Sample_num", axis=1).columns.tolist()
            sample_list = inp["Sample_num"].values
        else:
            gt = pd.read_csv(pp_prefix + "data/rna/peak_matrices/cellspecificSNRNA_ok.csv")
            inp = pd.read_csv(pp_prefix + "data/rna/peak_matrices/pseudobulkSNRNA_ok.csv")
            genes = inp.drop("Sample_num", axis=1).columns.tolist()
            gt = gt[gt.celltype != "None"]
            if ctrl_only:
                gt = gt[gt.individualID.isin(meta_ctrl.individualID)]
                inp = inp[inp.individualID.isin(meta_ctrl.individualID)]
            celltypes = gt["celltype"].sort_values().unique()
            print(celltypes)
            features = inp.iloc[:,1:].columns.tolist()
            nb_celltype = len(celltypes)
            nb_features = inp.shape[1] -1
            sample_list = inp[sampleid].values

            nb_sample = len(sample_list)
            X = np.zeros((nb_sample,
                            nb_celltype,
                            nb_features))
            for i,it in enumerate(sample_list):
                for j,cc in enumerate(celltypes):
                    tmp = gt[(gt.celltype==cc)
                                    &(gt[sampleid] ==it)][features].values
                    if len(tmp) ==0:
                        X[i,j,:] = np.zeros(X[i,j,:].shape)
                    else:
                        X[i, j, :] = gt[(gt.celltype==cc)
                                        &(gt[sampleid] ==it)][features].values
        if with_filter:
            mask = pd.read_csv(path_save + "mask_thrs_0.01.csv")
            mask.set_index("celltype", inplace=True)
            #mask = mask[features]
            mask = mask.loc[celltypes]
            X = X*np.asarray(mask.values)[np.newaxis,:,:]
        mix = inp.drop(sampleid, axis=1)
        if normalize:
            max_val= np.max(mix,1)
            max_val[max_val.values<1] = 1
            mix = mix/np.asarray(max_val)[:, np.newaxis]
            X = X/np.asarray(max_val)[:,np.newaxis, np.newaxis]
            mix = mix.values


        list_df= []
        list_df_genes= []
        cv_func = "KFold"

        if logo:
            cv_func = "LOGO"
            meta["Diagnosis"] = meta["cell_id"].str.split("_", expand=True)[1]
            tmp = inp.merge(meta, right_on="cell_id",
                    left_on="Sample_num")
            sample_list = tmp[groupby]

        elif with_real_bulk:
            bulk = bulk[bulk[sampleid].isin(sample_list)]
            sorter = sample_list
            sorterIndex = dict(zip(sorter, range(len(sorter))))
            bulk["index_order"] = bulk[sampleid].map(sorterIndex).values
            bulk.sort_values("index_order", ascending=True, inplace=True)
            bulk.drop("index_order", inplace=True, axis=1)


    #if True:
        if not os.path.exists(savename + "_nmf_per_genes.pickle"):
            if with_real_bulk:
                pred_, m_time = nmf_decomposition(
                                        bulk.drop(sampleid, axis=1),
                                        X)
                df_metrics_pr = compute_metrics_per_subject(pred_,
                                                X, celltypes,
                                                sample_list)
                df_metrics_genes = compute_metrics_per_genes(pred_,
                                                X, celltypes,
                                                genes)
            else:
                pred_, m_time = nmf_decomposition(
                                        mix,
                                        X)
                if with_filter:
                    pred_ = pred_*mask.values[np.newaxis,:,:]
                df_metrics_pr = compute_metrics_per_subject(pred_,
                                                X, celltypes,
                                                sample_list)
                print("NMF compute per genes...")
                df_metrics_genes = compute_metrics_per_genes(pred_,
                                                X, celltypes,
                                                genes)
            df_metrics_pr["method"] = "NMF"
            df_metrics_genes["method"] = "NMF"
            df_metrics_pr["time"] = m_time
            df_metrics_pr.to_pickle(savename + "_nmf_per_sub.pickle")
            df_metrics_genes.to_pickle(savename + "_nmf_per_genes.pickle")
            print("NMF mean time : " + str(m_time))
        else:
            df_metrics_pr = pd.read_pickle(savename + "_nmf_per_sub.pickle")
            df_metrics_genes = pd.read_pickle(savename + "_nmf_per_genes.pickle")

        list_df.append(df_metrics_pr)
        list_df_genes.append(df_metrics_genes)
        list_preds = [
                "LinearRegression",
                        ]
        for pr in list_preds:
            if not os.path.exists(savename + "_%s_per_it.pickle"%pr):
                if with_real_bulk:
                    (pred_,
                     mean_time) = multiOut_regression_Comp(
                                    mix,
                                    X,
                                sample_list,
                                 bulk.drop(sampleid,axis=1).values,
                                 X,
                                 sample_list,
                                    pr,
                                    K=5,
                                cv_func=cv_func)
                else:
                    (df_metrics_it, df_metrics_sub,
                    df_metrics_genes) = multiOut_regression(
                                    mix,
                                            X,
                                           sample_list,
                                            pr,
                                            celltypes,
                                            K=5,
                                            cv_func=cv_func)
                    if pr == "RandomForestRegressor":
                        np.save(savename + "pred_RF_nonmask.npy", pred_)
                        np.save(savename + "gt_RF_nonmask.npy", X)
                    if with_filter:
                        pred_ = pred_*mask.values[np.newaxis,:,:]
                        np.save(savename + "pred_RF.npy", pred_)

                #df_metrics_pr = compute_metrics_per_subject(pred_,
                #        X, celltypes,sample_list)
                print("%s compute per genes..."%pr)
                #df_metrics_genes = compute_metrics_per_genes(pred_,
                #                                X, celltypes,
                #                                genes)
                df_metrics_it["method"] = pr #+ "pseudobulk"
                df_metrics_sub["method"] = pr #+ "pseudobulk"
                df_metrics_genes["method"] = pr #+ "pseudobulk"
                #df_metrics_pr["time"] = mean_time
                #print(pr + "mean time: " + str(mean_time))
                df_metrics_sub.to_pickle(savename + "_%s_per_sub.pickle"%pr)
                df_metrics_it.to_pickle(savename + "_%s_per_it.pickle"%pr)
                df_metrics_genes.to_pickle(savename + "_%s_per_genes.pickle"%pr)
            else:
                df_metrics_sub = pd.read_pickle(savename + "_%s_per_sub.pickle"%pr)
                df_metrics_it = pd.read_pickle(savename + "_%s_per_it.pickle"%pr)
                df_metrics_genes = pd.read_pickle(savename + "_%s_per_genes.pickle"%pr)
            list_df.append(df_metrics_sub)
            list_df_genes.append(df_metrics_genes)
        df_metrics_tot = pd.concat(list_df, axis=0)
        df_metrics_tot_genes = pd.concat(list_df_genes, axis=0)
        df_metrics_tot.to_csv(savename + ".csv")
        df_metrics_tot_genes.to_csv(savename + "_genes.csv")
    else:
        df_metrics_tot = pd.read_csv(savename + ".csv")
        df_metrics_tot_genes = pd.read_csv(savename + "_genes.csv")
    palette = {
                "LinearRegression":"#AB6859",
                "NMF":"#AD3F25", #""#88352b",
                }
    hue_order=[ "NMF", "LinearRegression", ]
    list_df = []
    list_df.append(df_metrics_tot)
    dico_up = {
           # "Cellformer":"/remote/home/eloiseb/experiments/deconv_rna/berson_8k_map2_totnorm_lognorm_nosparse/"}
    "BayesPrism":"/remote/home/eloiseb/data/rna/bayesprism_/",
    "CIBERSORTx":"/home/eloiseb/data/rna/cibersort/cibersortx_",
            "Cellformer":"/remote/home/eloiseb/experiments/deconv_rna/universal_kfold_totnorm_log_nosparse/",
    }
            #"Cellformer_mask":"/home/eloiseb/experiments/deconv_rna/sepformer_ok_data_mask/_with_filter_/"}
    for k,v in dico_up.items():
        df_met = pd.read_csv(v + "metrics_all_sub.csv")
        df_met["method"] = k
        hue_order.append(k)
        palette[k] = "#576169"
        list_df.append(df_met)
    df_metrics_tot = pd.concat(list_df, axis=0)
    df_metrics_tot = df_metrics_tot[~df_metrics_tot.res.isna()]
    list_df = [df_metrics_tot_genes]
    for k,v in dico_up.items():
        df_met = pd.read_csv(v + "metrics_all_genes.csv")
        df_met["method"] = k
        list_df.append(df_met)
    df_metrics_tot_genes = pd.concat(list_df, axis=0)
    df_metrics_tot_genes = df_metrics_tot_genes[~df_metrics_tot_genes.res.isna()]
    df_metrics_tot_genes = df_metrics_tot_genes[df_metrics_tot_genes.res!=0]
    palette["BayesPrism"] = "#EBAFA1"
    palette["CIBERSORTx"] = "#eee9de"

    df_mse = df_metrics_tot[df_metrics_tot.metrics == "mse"]
   # df_mse["res"] = np.log(df_mse["res"].values)
    df_mse["metrics"] = "log_mse"
    #df_metrics_tot = pd.concat([df_metrics_tot, df_mse])

    df_mse = df_metrics_tot_genes[df_metrics_tot_genes.metrics == "mse"]
    #df_mse["res"] = np.log(df_mse["res"].values)
    df_mse["metrics"] = "log_mse"
    #df_metrics_tot_genes = pd.concat([df_metrics_tot_genes, df_mse])
    metrics = ["pearson","spearman", ]
    pairs=[((it, it), ("Cellformer", "Cellformer")) for it in hue_order if it !="Cellformer"]
    df_metrics_tot.res = df_metrics_tot.res.astype(float)
    df_metrics_tot_genes.res = df_metrics_tot_genes.res.astype(float)
    df_metrics_tot.loc[df_metrics_tot.fold.isna(), "fold"] = 0
    df_metrics_tot_genes.loc[df_metrics_tot_genes.fold.isna(), "fold"] = 0
    df_metrics_tot_genes= df_metrics_tot_genes.replace({"ENDO-Mural":"Endo-Mural"})
    df_metrics_tot= df_metrics_tot.replace({"ENDO-Mural":"Endo-Mural"})
    ###Plot model comparison
    plot_model_comparison(df_metrics_tot,
                        savename,
                        palette,
                        pairs,
                        metrics=metrics,
                        hue_order=hue_order)
    celltypes = ["AST", "Endo-Mural", "EXC",
                     "INH", "MIC", "OLD",
                        "OPC"]
    pairs=[((cct,it), (cct, "Cellformer")) for it in hue_order if it !="Cellformer" for cct in celltypes]
    plot_model_comparison_stratified_ct(df_metrics_tot,
                        savename,
                        palette,
                        pairs,
                        metrics=metrics,
                        hue_order=hue_order)
    ###plot celltype
    plot_per_celltype(df_metrics_tot,
                    "Cellformer",
                    savename,
                    metrics=metrics,
                    )
    pairs=[((it, it), ("Cellformer", "Cellformer")) for it in hue_order if it !="Cellformer"]
    plot_comparison_per_genes(df_metrics_tot_genes,
                            savename,
                            palette,
                            pairs,
                        metrics=metrics,
                        hue_order=hue_order)
    plot_gene_per_celltype(df_metrics_tot_genes,
                    "Cellformer",
                    savename,
                    metrics=metrics,
                    )




    ###Per genes

if __name__ == "__main__":
    main()







