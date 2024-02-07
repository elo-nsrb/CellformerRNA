import os
import random
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import torch.nn as nn 

import asteroid
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.data.wsj0_mix import Wsj0mixDataset
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device

from asteroid.models import DPRNNTasNet
from my_data import *
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc,precision_recall_curve, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from test_functions import *
from analysis import *
from utils import get_logger, parse #device, 
from src_ssl.models import *
from src_ssl.models.sepformer_tasnet import SepFormerTasNet, SepFormer2TasNet
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_mse, pairwise_mse, singlesrc_neg_sisdr
from my_losses import singlesrc_bcewithlogit, combinedpairwiseloss, combinedsingleloss, fpplusmseloss
from my_network import *
import sys
sys.path.append("../")
from ML_models import *

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet", "FC_MOR", "NL_MOR"])
parser.add_argument("--gpu", default="2")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--out_dir", type=str, default="results/best_model", help="Directory in exp_dir where the eval results will be stored")
parser.add_argument('--peak_count_matrix', type=str,
                        default="/home/eloiseb/stanford_drive/data/ATAC-seq/count_from_sc_peaks/AD/CTRL/CTRL_CAUD_AD.peak_countMatrix.txt",
                        help='Bulk sample')

parser.add_argument('--groundtruth',
                        default=None,
                        help='Bulk sample')
parser.add_argument('--type', type=str,
                        default="bulk",
                        )

parser.add_argument("--ckpt_path", default="best_model.pth", help="Experiment checkpoint path")
parser.add_argument('--binarize', action='store_true',
                        help='binarize output')
parser.add_argument('--mask', action='store_true',
                        help='binarize output')
parser.add_argument('--pure', action='store_true',
                        help='Test on pure')
parser.add_argument('--save', action='store_true',
                        help='Test on pure')



def main(args):
    parent_dir = args.parent_dir
    res_cv_raw = None
    df_metrics_per_sub_list = []
    df_metrics_per_it_list = []
    df_metrics_per_gene_list = []
    gt_m = None
    for s_id in np.arange(5):
        args.model_path = os.path.join(parent_dir,
                                    "exp_kfold_%s/"%(s_id))
        model_path = os.path.join(args.model_path, args.ckpt_path)
        savedir = os.path.join(args.model_path, args.type)
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        if args.ckpt_path != "best_model.pth":
            savedir = os.path.join(savedir,
                    args.ckpt_path.split("/")[-1].split(".")[0])
            if not os.path.exists(savedir):
                os.mkdir(savedir)

        opt = parse(args.model_path + "train.yml", is_tain=True)
        testset = opt["datasets"]["sample_id_test"]
        if args.type == "bulk":
            mixtures = pd.read_csv(args.peak_count_matrix, sep="\t",header=1)
            name = args.peak_count_matrix.split("/")[-1].split(".")[0]
            print(name)
            #mixtures = mixtures[mixtures.Geneid.isin(annot.Geneid)]
            mixtures = mixtures.iloc[:,6:].T
            mixtures_tt["Sample_num"] = mixtures_tt.index.tolist()
            mixtures["Sample_num"] = mixtures.index.tolist()
            savedir += "/bulk_sample_decon/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        elif args.type == "pseudobulk":
            mixtures = pd.read_csv(args.peak_count_matrix, index_col=0)
            name = args.peak_count_matrix.split("/")[-1].split(".")[0]
            print(name)
            if "Sample_num" not in mixtures.columns.tolist():
                mixtures["Sample_num"] = mixtures.index.tolist()
                mixtures.reset_index(inplace=True, drop=True)
            savedir = os.path.join(savedir, name)
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            savedir += "/pseudobulk_sample_decon/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        print("Savedir : " + savedir)
        celltypes = opt["datasets"]["celltype_to_use"]
        num_spk = len(celltypes)
        if args.ckpt_path == "best_model.pth":
            # serialized checkpoint
            if args.model == "FC_MOR":
                model = MultiOutputRegression(**opt["net"])
                model = model.from_pretrained(model_path, **opt["net"])
            elif args.model == "NL_MOR":
                model = NonLinearMultiOutputRegression(**opt["net"])
                model = model.from_pretrained(model_path, **opt["net"])
            else:
                model = getattr(asteroid.models, args.model).from_pretrained(model_path)
        else:
            model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])
            all_states = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
            model.load_state_dict(state_dict)

        if args.use_gpu:
            model.cuda()
        model_device = next(model.parameters()).device
        series_list = []
        binary = False
        perfs = []
        torch.no_grad().__enter__()
        if opt['datasets']["normalizeMax"]:
            max_mix = mixtures.drop("Sample_num", axis=1).max(1) 
            mixtures_i = mixtures.drop("Sample_num",axis=1)/max_mix[:,np.newaxis]
        mix = torch.from_numpy(mixtures_i.values.astype(
                                             np.float32))
        mix = tensors_to_device(mix, device=model_device)

        sources_res = model(mix)
        sources_res = torch.clamp(sources_res, min=0)
        sources_res_np = sources_res.detach().cpu().numpy()
        if opt['datasets']["normalizeMax"]:
            sources_res_np = sources_res_np*max_mix[:, 
                                    np.newaxis, np.newaxis]
            mix = mix.detach().cpu().numpy()*max_mix[:, np.newaxis]
        if res_cv_raw is None:
            res_cv_raw = np.zeros_like(sources_res_np)
            gt_m = np.zeros_like(sources_res_np)
        test_sp = sources_res_np[mixtures["Sample_num"].isin(testset)]
        testset_idx = mixtures[mixtures.Sample_num.isin(testset)].index.values.tolist() 
        trainset_idx = mixtures[~mixtures.Sample_num.isin(testset)].index.values.tolist() 
        res_cv_raw[testset_idx,:,:] = test_sp
        df_res = [] 
        for k, ct in enumerate(celltypes):
            df_ = pd.DataFrame(sources_res_np[:,k,:], 
                                index=mixtures.index, 
                                columns=mixtures.columns.tolist()[:-1])
            df_["celltype"] = ct
            df_res.append(df_)
        df_res_f = pd.concat(df_res, axis=0)
        df_res_f.to_csv(savedir + name +".csv")

        if args.groundtruth is not None:
            print(savedir)
            separate = np.load(args.groundtruth)["mat"]
            if len(separate.shape)<2:
                separate = np.expand_dims(separate, 0)
            print(separate.shape)
            _, separate = gatherCelltypes(celltypes, 
                                        separate, opt["datasets"]["celltypes"] )
            gt_sp = separate[mixtures["Sample_num"].isin(testset)]
            df_met_per_it = compute_metrics(test_sp, 
                            gt_sp,
                            celltypes)
            df_met_per_it["fold"] = "fold_%s"%str(s_id)
            df_metrics_per_it_list.append(df_met_per_it)
            df_met_per_sub = compute_metrics_per_subject(test_sp,
                            gt_sp,
                            celltypes,
                            testset) 
            df_met_per_sub["fold"] = "fold_%s"%str(s_id)
            df_metrics_per_sub_list.append(df_met_per_sub)
            #df_met_per_gene = compute_metrics_per_genes(test_sp,
            #                gt_sp,
            #                celltypes,
            #                np.arange(gt_sp.shape[-1])) 
            #df_met_per_gene["fold"] = "fold_%s"%str(s_id)
            #df_metrics_per_gene_list.append(df_met_per_gene)
    df_metrics_per_it = pd.concat(df_metrics_per_it_list,axis=0)
    df_metrics_per_sub = pd.concat(df_metrics_per_sub_list,axis=0)
    #df_metrics_per_gene = pd.concat(df_metrics_per_gene_list,axis=0)
    df_metrics_per_it.to_csv(os.path.join(parent_dir,name + "ALL_runs_metrics_cv_per_itt.csv"))
    df_metrics_per_sub.to_csv(os.path.join(parent_dir,name + "ALL_runs_metrics_cv_per_subt.csv"))
    #df_metrics_per_gene.to_csv(os.path.join(parent_dir,name + "ALL_runs_metrics_cv_per_genet.csv"))
    print("Mean spearman per it : %s"%str(df_metrics_per_it[df_metrics_per_it.metrics=="spearman"]["res"].mean()))
    print("Mean spearman per sub : %s"%str(df_metrics_per_sub[df_metrics_per_sub.metrics=="spearman"]["res"].mean()))
   # print("Mean spearman per gene : %s"%str(df_metrics_per_gene[df_metrics_per_gene.metrics=="spearman"]["res"].mean()))
    __import__('ipdb').set_trace()


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)
