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
from asteroid.losses import *

from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device

from data import make_dataloader,gatherCelltypes
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc,precision_recall_curve, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from utils import get_logger, parse #device,
from src_ssl.models import *
from src_ssl.models.sepformer_tasnet import SepFormerTasNet, SepFormer2TasNet
import sys
sys.path.append("../")
from comparison_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet", "FC_MOR", "NL_MOR"])
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
    opt_p = parse(os.path.join(args.parent_dir, "train.yml"), is_tain=True)
    if hasattr(opt_p["datasets"], "gene_filtering") :
        if not opt_p["datasets"]["gene_filtering"] is None:
            print("masking")
            mask = pd.read_csv(opt_p["datasets"]["gene_filtering"])
            mask.set_index("celltype", inplace=True)
    list_files = glob.glob(os.path.join(parent_dir, "exp_kfold_*"))
    for s_id in np.arange(len(list_files)):
    #if True:
    #    s_id=2
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
        #if True:
        if not os.path.exists(os.path.join(savedir, "metrics_genes.csv")):
            opt = parse(args.model_path + "train.yml", is_tain=True)
            celltypes = opt["datasets"]["celltype_to_use"]
            num_spk = len(celltypes)

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
                    print(args.ckpt_path)
                    state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
                    model.load_state_dict(state_dict)
                    # model.load_state_dict(all_states["state_dict"], strict=False)

                # Handle device placement
                if args.use_gpu:
                    model.cuda()
                model_device = next(model.parameters()).device
                print(model)
                if args.testset == "train":
                    use_train = True
                else:
                    use_train = False
                val_loader = make_dataloader("test",
                                     is_train=False,
                                        data_kwargs=opt['datasets'],
                                        num_workers=opt['datasets'] ['num_workers'],
                               batch_size=opt["training"]["batch_size"],
                                           limit=limit,
                                           use_train=use_train,
                                           custom_testset=args.custom_testset,
                                           pure=args.pure)#.data_loader
        #
                series_list = []
                binary = False
                perfs = []
                torch.no_grad().__enter__()

                if limit is None:
                    limit=len(val_loader)
                #for idx in tqdm(range(mixtures.shape[0])):
                mixture_ = []
                separate_ = []
                separate_pred = []
                label_ = []
                for i, batch in enumerate(val_loader):
                #if i*len(batch)<limit:
                    with torch.no_grad():
                        mix = batch[0]
                        sources = batch[1]
                        label = batch[2]
                    # Forward the network on the mixture.
                        mix = tensors_to_device(mix, device=model_device)
                        sources = tensors_to_device(sources, device=model_device)
                        est_sources = model(mix)
                        sig = [0]*len(celltypes)
                        mix_np = mix.cpu().data.numpy()
                        sources_np = sources.cpu().data.numpy()
                        est_sources_np = est_sources.cpu().data.numpy()
                        if args.masking:
                            est_sources_np = est_sources_np*bin_matrix #mask
                        mixture_.append(mix_np)
                        separate_.append(sources_np)
                        separate_pred.append(est_sources_np)
                        label_ += label
                mixtures = np.concatenate(mixture_)
                separates = np.concatenate(separate_)
                separates_pred = np.concatenate(separate_pred)
               # if not val_loader.dataset.gene_filtering is None:
                if hasattr(opt_p["datasets"], "gene_filtering") :
                    if not opt_p["datasets"]["gene_filtering"] is None:
                        separates = separates*mask.values[np.newaxis, :,:]
                        separates_pred = separates_pred*mask.values[np.newaxis, :,:]
                del model
                torch.cuda.empty_cache()
                name = args.testset + "_" + str(args.pure)
                if args.custom_testset is not None:
                    name += args.custom_testset
                if args.masking:
                    name += "_masking_with_input_"
                if args.save:
                    print("Saving ...")
                    labels = np.concatenate(label_)
                    ppp = parent_dir.split("/")[-2]
                    print(ppp)
                    tmp = os.path.join("/home/eloiseb/experiments/deconv_rna/",
                                        ppp)
                    tmp = os.path.join(tmp,
                                        "exp_kfold_%s/"%(s_id))
                    if not os.path.exists(tmp):
                        os.mkdir(tmp)
                    np.savez_compressed(os.path.join(tmp, "test_predictions_"
                                                    + name+ ".npz"),
                                        mat=separates_pred)
                    np.savez_compressed(os.path.join(tmp, "label_"
                                                    + name+ ".npz"),
                                        mat=labels)
                    np.savez_compressed(os.path.join(tmp, "test_true_"
                                                    + name+ ".npz"),
                                        mat=separates)
                    np.savez_compressed(os.path.join(tmp, "test_mixtures_"
                                                    + name+ ".npz"),
                                        mat=mixtures)
                    np.savez_compressed(os.path.join(args.model_path, "predictions_"
                                                    + name+ ".npz"),
                                        mat=separates_pred)
                    np.savez_compressed(os.path.join(args.model_path, "true_"
                                                    + name+ ".npz"),
                                        mat=separates)
                    np.savez_compressed(os.path.join(args.model_path, "mixtures_"
                                                    + name+ ".npz"),
                                        mat=mixtures)
                print(opt['datasets']["hdf_dir"])
                os.remove(os.path.join(opt['datasets']["hdf_dir"],
                                    "test.hdf5"))


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
            df_metrics_per_subject.to_csv(os.path.join(savedir, "metrics_subjects.csv"))
            df_metrics_per_it.to_csv(os.path.join(savedir, "metrics_it.csv"))
            df_metrics_per_genes.to_csv(os.path.join(savedir, "metrics_genes.csv"))
        else:
            df_metrics_per_subject= pd.read_csv(os.path.join(savedir, "metrics_subjects.csv"))
            df_metrics_per_it = pd.read_csv(os.path.join(savedir, "metrics_it.csv"))
            df_metrics_per_genes = pd.read_csv(os.path.join(savedir, "metrics_genes.csv"))

        df_metrics_sub_list.append(df_metrics_per_subject)
        df_metrics_it_list.append(df_metrics_per_it)
        df_metrics_genes_list.append(df_metrics_per_genes)
    if hasattr(opt_p["datasets"], "gene_filtering") :
        if not val_loader.dataset.gene_filtering is None:
            args.parent_dir += "_with_filter_"
            if not os.path.exists(args.parent_dir):
                os.mkdir(args.parent_dir)
    df_metrics = pd.concat(df_metrics_it_list)
    df_metrics.to_csv(os.path.join(args.parent_dir,
                        "metrics_all_it.csv"),
                    index=None)
    df_metrics = pd.concat(df_metrics_sub_list)
    df_metrics.to_csv(os.path.join(args.parent_dir,
                        "metrics_all_sub.csv"),
                    index=None)
    df_metrics = pd.concat(df_metrics_genes_list)
    df_metrics.to_csv(os.path.join(args.parent_dir,
                        "metrics_all_genes.csv"),
                    index=None)
    __import__('ipdb').set_trace()



if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)
