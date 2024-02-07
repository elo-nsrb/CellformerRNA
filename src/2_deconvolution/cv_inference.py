import os
import glob
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
from asteroid.losses import *

from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device

from asteroid.models import DPRNNTasNet
from my_data import make_dataloader,gatherCelltypes
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc,precision_recall_curve, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from test_functions import *
from utils import get_logger, parse #device, 
from src_ssl.models import *
from src_ssl.models.sepformer_tasnet import SepFormerTasNet, SepFormer2TasNet
import sys
sys.path.append("../")
from ML_models import *

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet", "FC_MOR", "NL_MOR"])
parser.add_argument('--peak_count_matrix', type=str,
                        help='Bulk sample')

parser.add_argument('--type', type=str,
                        default="pseudobulk",
                        )
parser.add_argument('--groundtruth',
                        default=None,
                        help='cell type-specific samples')
parser.add_argument("--gpu", default="2")
parser.add_argument("--ckpt_path", default="best_model.pth", help="Experiment checkpoint path")
parser.add_argument("--resume", action="store_true", help="Resume-training")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--out_dir", type=str, default="results/best_model", help="Directory in exp_dir where the eval results will be stored")
parser.add_argument('--testset', default="test",
                        help='partition to use')
parser.add_argument('--limit', default=None)
parser.add_argument('--custom_testset', default=None,
                        help='Use a custom testset')


parser.add_argument('--binarize', action='store_true',
                        help='binarize output')
parser.add_argument('--masking', action='store_true',
                        help='binarize output')
parser.add_argument('--pure', action='store_true',
                        help='Test on pure')
parser.add_argument('--save', action='store_true',
                        help='Test on pure')



def main(args):
    parent_dir = args.parent_dir
    df_metrics_sub_list = []
    df_metrics_it_list = []
    df_metrics_genes_list = []
    gt_m = None
    opt_p = parse(args.parent_dir + "train.yml", is_tain=True)
    celltype = opt_p["datasets"]["celltype_to_use"]
    list_files = glob.glob(os.path.join(parent_dir, "exp_kfold_*"))
    for s_id in np.arange(len(list_files)):
        mixtures_all = pd.read_csv(args.peak_count_matrix)#, index_col=0)
       # mixtures_all = mixtures_all.reset_index()
        args.model_path = os.path.join(parent_dir,
                                    "exp_kfold_%s/"%(s_id))
        print(args.model_path)
        model_path = os.path.join(args.model_path, args.ckpt_path)
        savedir = os.path.join(args.model_path, args.testset )
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        if args.ckpt_path != "best_model.pth":
            savedir = os.path.join(savedir,
                    args.ckpt_path.split("/")[-1].split(".")[0])
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        if ("gene_filtering" in opt_p["datasets"]) and not (opt_p["datasets"]["gene_filtering"] is None):
            print("masking")
            mask = pd.read_csv(opt_p["datasets"]["gene_filtering"])
            mask.set_index("celltype", inplace=True)
        if args.type == "bulk":
            mixtures = pd.read_csv(args.peak_count_matrix)#, sep="\t",header=1)
            mixtures.rename({"Unnamed: 0":"Sample_num"}, axis=1, inplace=True)
            savedir += "/bulk_sample_decon_%s/"%args.peak_count_matrix.split("/")[-1].split(".")[0]
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        elif args.type == "pseudobulk":
                mixtures_all = pd.read_csv(args.peak_count_matrix, index_col=0)
                mixtures_all = mixtures_all.reset_index()
        opt = parse(args.model_path + "train.yml", is_tain=True)
        sample_id_test = opt["datasets"]["sample_id_test"]
        celltypes = opt["datasets"]["celltype_to_use"]
        num_spk = len(celltypes)
        if args.type == "bulk":
            savedir += "/bulk_sample_decon/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        elif args.type == "pseudobulk":
            print(sample_id_test)
            mixtures = mixtures_all[mixtures_all["Sample_num"].isin(sample_id_test)]
            index_keep = mixtures.index
            savedir += "/pseudobulk_sample_decon/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        label_ = mixtures.Sample_num.tolist()
        print("Savedir : " + savedir)
    

        # all resulting files would be saved in eval_save_dir
        eval_save_dir = os.path.join(args.model_path, args.out_dir)
        if args.custom_testset is not None:
            eval_save_dir = os.path.join(eval_save_dir, args.custom_testset)
        os.makedirs(eval_save_dir, exist_ok=True)

        #limit = 1000
        if args.limit is not None:
            limit = eval(args.limit)
        else:
            limit = None
        if not os.path.exists(os.path.join(eval_save_dir, "final_metrics.json")):
            if args.ckpt_path == "best_model.pth":
                # serialized checkpoint
                model = getattr(asteroid.models, args.model).from_pretrained(model_path)
                print(model_path)
            else:
                # non-serialized checkpoint, _ckpt_epoch_{i}.ckpt, keys would start with
                # "model.", which need to be removed
                model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])
                all_states = torch.load(args.ckpt_path, map_location="cpu")
                state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
                model.load_state_dict(state_dict)
                # model.load_state_dict(all_states["state_dict"], strict=False)

            # Handle device placement
            if args.use_gpu:
                model.cuda()
            model_device = next(model.parameters()).device
            if args.testset == "train":
                use_train = True
            else:
                use_train = False
            series_list = []

            with torch.no_grad():
                mix = mixtures.drop("Sample_num", 
                                        axis=1).values.astype(
                                             np.float32)
                if opt['datasets']["normalizeMax"]:
                    max_val= np.max(mix,1)
                    mix = mix/max_val[:, np.newaxis]
                mix = torch.from_numpy(mix)
                mix = tensors_to_device(mix, device=model_device)
                est_sources = model(mix)
                mix_np = mix.cpu().data.numpy()
                est_sources_np = est_sources.cpu().data.numpy()
                if opt['datasets']["normalizeMax"]:
                    est_sources_np = est_sources_np*max_val[:, np.newaxis,np.newaxis]
            if args.save:
                print("Saving ...")
                ppp = parent_dir.split("/")[-2]
                print(ppp)
                tmp = os.path.join("/home/eloiseb/experiments/deconv_rna/",
                                    ppp)
                tmp = os.path.join(tmp,
                                    "exp_kfold_%s/"%(s_id))
                tmp = savedir
                if not os.path.exists(tmp):
                    os.mkdir(tmp)
                #np.savez_compressed(os.path.join(args.model_path,
                np.savez_compressed(os.path.join(tmp,
                                            "predictions_pseudobulk_Test_no_filt" 
                                                + ".npz"),
                                    mat=est_sources_np)
                np.savez_compressed(os.path.join(tmp,
                                            "labels_Test" 
                                                + ".npz"),
                                    mat=np.asarray(label_))
            if ("gene_filtering" in opt_p["datasets"]) and not (opt_p["datasets"]["gene_filtering"] is None):
                mask = pd.read_csv(opt_p["datasets"]["gene_filtering"])
                mask.set_index("celltype", inplace=True)
                separates_pred = est_sources_np*mask.values[np.newaxis, :,:]
            else:
                separates_pred = est_sources_np
            del model
            torch.cuda.empty_cache()
            if args.save:
                ppp = parent_dir.split("/")[-2]
                print(ppp)
                tmp = os.path.join("/home/eloiseb/experiments/deconv_rna/",
                                    ppp)
                tmp = os.path.join(tmp,
                                    "exp_kfold_%s/"%(s_id))
                np.savez_compressed(os.path.join(tmp,
                                            "predictions_pseudobulk_Test" 
                                                + ".npz"),
                                    mat=separates_pred)
        

        if args.groundtruth is not None:
            print(savedir)
            gt = pd.read_csv(args.groundtruth)
            sampleid = "Sample_num"
            nb_sample = len(label_)

            X = np.zeros(separates_pred.shape)
            for i,it in enumerate(label_):
                for j,cc in enumerate(celltype):
                    tmp = gt[(gt.celltype==cc)
                                    &(gt[sampleid] ==it)]
                    if len(tmp) ==0:
                        X[i,j,:] = np.zeros(X[i,j,:].shape)
                    else:
                        X[i, j, :] = tmp[(gt.celltype==cc)
                                    &(tmp[sampleid] ==it)].drop(
                                            ["Sample_num",
                                                "celltype"], axis=1).values
            separates = X
            if ("gene_filtering" in opt_p["datasets"]) and not (opt_p["datasets"]["gene_filtering"] is None):
                separates = separates#*mask.values[np.newaxis, :,:]
            if separates.shape[0]>1:
                df_metrics_per_subject= compute_metrics_per_subject(separates_pred, 
                                    separates, 
                                    celltypes,
                                    label_) 
                df_metrics_per_subject["fold"] = "fold_%s"%str(s_id)
                df_metrics_per_it= compute_metrics(separates_pred, 
                                    separates, 
                                    celltypes)
                df_metrics_per_it["fold"] = "fold_%s"%str(s_id)
                df_metrics_per_genes= compute_metrics_per_genes(separates_pred, 
                                    separates, 
                                    celltypes,
                                    list(np.arange(separates.shape[-1]))) 
                df_metrics_per_genes["fold"] = "fold_%s"%str(s_id)
                #mixtures = pd.DataFrame 
            df_metrics_per_subject.to_csv(os.path.join(savedir, "metrics_per_subjects.csv"))
            df_metrics_per_it.to_csv(os.path.join(savedir, "metrics_per_it.csv"))
            df_metrics_sub_list.append(df_metrics_per_subject)
            df_metrics_it_list.append(df_metrics_per_it)
            df_metrics_genes_list.append(df_metrics_per_genes)
    if args.groundtruth is not None:
        if not opt_p["datasets"]["gene_filtering"] is None:
            args.parent_dir += "_with_filter_"
            print(args.parent_dir)
            if not os.path.exists(args.parent_dir):
                os.mkdir(args.parent_dir)
        df_metrics = pd.concat(df_metrics_it_list)
        df_metrics.to_csv(os.path.join(args.parent_dir, 
                            "PSEUDOBULK_metrics_all_per_it.csv"),
                        index=None)
        df_metrics = pd.concat(df_metrics_sub_list)
        df_metrics.to_csv(os.path.join(args.parent_dir, 
                            "PSEUDOBULK_metrics_all_per_sub.csv"),
                        index=None)
        df_metrics = pd.concat(df_metrics_genes_list)
        df_metrics.to_csv(os.path.join(args.parent_dir, 
                            "PSEUDOBULK_metrics_all_per_genes.csv"),
                        index=None)
        __import__('ipdb').set_trace()



if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)
