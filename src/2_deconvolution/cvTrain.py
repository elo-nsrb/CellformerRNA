#steroid is based on PyTorch and PyTorch-Lightning.
import comet_ml
import torch
import torch.nn as nn
import asteroid

from torch import optim
from utils import get_logger, parse, save_opt #device,
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from schedulers import DPTNetScheduler
import pytorch_lightning as pl
from src_ssl.models import *
import json
import yaml
from network import *
from sklearn.model_selection import LeaveOneGroupOut, KFold, LeaveOneOut
from losses import *
from data import *
import argparse
from engine import System
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="SepFormerTasNet", choices=[ "SepFormerTasNet"])
parser.add_argument("--gpu", default="2")
parser.add_argument("--resume_ckpt", default="last.ckpt", help="Checkpoint path to load for resume-training")
parser.add_argument("--resume", action="store_true", help="Resume-training")



def main(args):
    seed_everything(42, workers=True)
    parent_dir = args.parent_dir
    opt = parse(os.path.join(parent_dir, "train.yml"), is_tain=True)
    list_ids = []
    with open(os.path.join(opt["datasets"]["sample_list_file"]), "r") as f:
        list_ids = f.read().splitlines()
    list_ids = list(set(list_ids))
    list_ids.sort()
    cv_func = opt["datasets"]["cv_func"]
    groupby = opt["datasets"]["groupby"]
    if cv_func =="kfold":
        kf = KFold(n_splits=opt["datasets"]["k"], shuffle=True, random_state=23)
        kf.get_n_splits(list(set(list_ids)))
        list_splits = kf.split(list_ids)
    elif cv_func == "logo":
        if (groupby=="sample"):
            logo =  LeaveOneOut()
            list_splits = logo.split(list_ids)
        elif ((opt["datasets"]["name"] == "pbmc") & (groupby=="Method")):
            logo =  LeaveOneGroupOut()
            groups = [it.split("_")[1].split("_")[0] for it in list_ids]
            list_splits = logo.split(list_ids, groups=groups)
        elif (opt["datasets"]["name"] != "pbmc"):
            logo =  LeaveOneGroupOut()
            list_diagnosis = [it.split("_")[1] for it in list_ids]
            #meta = pd.read_csv("/home/eloiseb/data/rna/pseudobulks_sum/cellinfo.csv")
            #meta["brain_region"] = meta["cell_id"].str.rsplit("_",
            #                                2, expand=True)[2]
            mapping = {"MTG":"SMTG", "NAC":"NAC", "SMTG":"SMTG", "MFG":"MFG", "CTX":"CTX", "PCTX":"CTX", "DLFC":"CTX",
                    "SN":"SN", "Cortex":"CTX", "DG":"HIPP","CA1":"HIPP", "HIPP":"HIPP", "CA24":"HIPP","DLPFC":"DLPFC",
                    "EC":"EC", "SUB":"HIPP", "AMY":"AMY", "SACC":"SACC"}
            list_region = [mapping[it.split("_")[2]] for it in list_ids]
            #meta["brain_region"] = meta["brain_region"].map(mapping).values
            #meta.drop("cell_type", axis=1, inplace=True)
            #meta.drop("cell_subtype", axis=1, inplace=True)
            #meta = meta[~meta.duplicated()]
            #meta = meta[meta.cell_id.isin(list_ids)]
            #meta["Diagnosis"] = meta["cell_id"].str.split("_", expand=True)[1]
            #groups = meta[opt["datasets"]["groupby"]].values.tolist()
            if groupby == "Diagnosis":
                groups = list_diagnosis

            elif groupby=="Region":
                groups=list_region
            else:
                raise NotImplemented("Please use Diagnosis or Region as a grouping variable")
            list_splits = logo.split(list_ids, groups=groups)
    else:
        raise "Cross validation function not implemented"
    for i, (train_id, test_id) in enumerate(list_splits):
        #if i>0:
                s_id = np.asarray(list_ids)[test_id].tolist()
                opt = parse(os.path.join(parent_dir , "train.yml"), is_tain=True)
                opt["datasets"]["sample_id_test"] = s_id
                opt["datasets"]["sample_id_val"] = None
                opt["datasets"]["hdf_dir"] = os.path.join(
                                    opt["datasets"]["hdf_dir"],
                                            "hdfho_kfold_%s/"%(str(i)))#, val_id))
                print(opt["datasets"]["hdf_dir"])
                #print(s_id)
                args.model_path = os.path.join(parent_dir,
                                            "exp_kfold_%s/"%(str(i)))#,val_id))
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                save_opt(opt, args.model_path + "train.yml")
                logger = get_logger(__name__)
                logger.info('Building the model of %s'%args.model)
                train_loader = make_dataloader("train",
                                                is_train=True,
                                data_kwargs=opt['datasets'],
                                num_workers=opt['datasets']['num_workers'],
                               batch_size=opt["training"]["batch_size"],
                               ratio=opt['datasets']["ratio"])#.data_loader
                val_loader = make_dataloader("val",is_train=True,
                                            data_kwargs=opt['datasets'],
                                            num_workers=opt['datasets'] ['num_workers'],
                                            batch_size=opt["training"]["batch_size"],
                                            ratio=opt['datasets']["ratio"])


                n_src = len(opt["datasets"]["celltype_to_use"])

                if args.model == "FC_MOR":
                    model = MultiOutputRegression(**opt["net"])
                    print(model)
                elif args.model == "NL_MOR":
                    model = NonLinearMultiOutputRegression(**opt["net"])
                    print(model)
                else:
                    model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])

                learnable_params = list(model.parameters())
                if opt["training"]["loss"] == "mse":
                    loss = nn.MSELoss()
                elif opt["training"]["loss"] == "mse_weighted":
                    #loss = nn.MSELoss()
                    loss = weightedloss(src=n_src, method="Uncertainty")
                    learnable_params += list(loss.parameters())
                elif opt["training"]["loss"] == "bce":
                    loss = nn.BCEWithLogitsLoss()
                elif opt["training"]["loss"] == "mse_no_pit":
                    #loss = nn.MSELoss()
                    if opt["training"]["weights"] is not None:
                        weights = np.asarray(opt["training"]["weights"])
                    else:
                        weights = None
                    loss = weightedloss(src=n_src, weights=weights)
                elif opt["training"]["loss"] == "mask_mse":
                    mask = train_loader.dataset.mask.values[np.newaxis, :,:]
                    loss = MaskMSE(mask)
                elif opt["training"]["loss"] == "combined":
                    loss = CombinedSingleFunction()
                elif opt["training"]["loss"] == "bcemse":
                    loss = BCEMSE_loss()
                elif opt["training"]["loss"] == "fp_mse":
                    loss = fpplusmseloss

                optimizer = optim.AdamW(learnable_params, lr=1e-3)
                # Define scheduler
                scheduler = None
                #if args.model in ["DPTNet", "SepFormerTasNet", "SepFormer2TasNet"]:
                #    steps_per_epoch = len(train_loader) // opt["training"]["accumulate_grad_batches"]
                #    opt["scheduler"]["steps_per_epoch"] = steps_per_epoch
                #    scheduler = {
                #                "scheduler": DPTNetScheduler(
                #                optimizer=optimizer,
                #                steps_per_epoch=steps_per_epoch,
                #                 d_model=model.masker.mha_in_dim,
                #                 ),
                #                     "interval": "batch",
                #                    }
                #if opt["training"]["reduce_on_plateau"]:
                if True:
                        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                                      factor=0.8,
                                                      patience=2,
                                                      )

                system = System(model, optimizer, loss, train_loader, val_loader,
                        scheduler=scheduler)


                # Define callbacks
                callbacks = []
                checkpoint_dir = os.path.join(args.model_path, "checkpoints/")
                if opt["datasets"]["only_training"]:
                        monitor = "loss"
                else:
                        monitor = "val_loss"
                checkpoint = ModelCheckpoint(dirpath=checkpoint_dir,
                                        filename='{epoch}-{step}',
                                        monitor=monitor, mode="min",
                                    save_top_k=opt["training"]["save_epochs"],
                                    save_last=True, verbose=True,
                                     )
                callbacks.append(checkpoint)
                if opt["training"]["early_stop"]:

                    callbacks.append(EarlyStopping(monitor=monitor,
                                        mode="min",
                                        patience=opt["training"]["patience"],
                                        verbose=True,
                                        min_delta=0.0))
                lr_monitor = LearningRateMonitor(logging_interval='step')
                callbacks.append(lr_monitor)
                loggers = []
                tb_logger = pl.loggers.TensorBoardLogger(
                os.path.join(args.model_path, "tb_logs/"),
                )
                loggers.append(tb_logger)
                if opt["training"]["comet"]:
                    comet_logger = pl.loggers.CometLogger(
                        save_dir=os.path.join(args.model_path, "comet_logs/"),
                        experiment_key=opt["training"].get("comet_exp_key", None),
                        log_code=True,
                        log_graph=True,
                        parse_args=True,
                        log_env_details=True,
                        log_git_metadata=True,
                        log_git_patch=True,
                        log_env_gpu=True,
                        log_env_cpu=True,
                        log_env_host=True,
                        )
                    comet_logger.log_hyperparams(opt)
                    loggers.append(comet_logger)



                # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
                # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
                if args.resume:
                    resume_from = os.path.join(checkpoint_dir, args.resume_ckpt)
                else:
                    resume_from = None
                trainer = Trainer(max_epochs=opt["training"]["max_epochs"],
                                #batch_size =opt["training"]["batch_size"],
                                logger=loggers,
                                callbacks=callbacks,
                                default_root_dir=args.model_path,
                        accumulate_grad_batches=opt[ "training"]["accumulate_grad_batches"],
                            #resume_from_checkpoint=resume_from,
                                #deterministic=True,
                        accelerator="gpu",
                                devices=2,
                                )
                trainer.fit(system, ckpt_path=resume_from)

                best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
                with open(os.path.join(args.model_path, "best_k_models.json"), "w") as f:
                        json.dump(best_k, f, indent=0)

                state_dict = torch.load(checkpoint.best_model_path)
                system.load_state_dict(state_dict=state_dict["state_dict"])
                system.cpu()

                train_set_infos = dict()
                train_set_infos["dataset"] = "Brain"

                to_save = system.model.serialize()
                to_save.update(train_set_infos)
                torch.save(to_save, os.path.join(args.model_path, "best_model.pth"))
                print(opt['datasets']["hdf_dir"])
                os.remove(os.path.join(opt['datasets']["hdf_dir"],
                                "train.hdf5"))
                os.remove(os.path.join(opt['datasets']["hdf_dir"],
                                "val.hdf5"))
                torch.cuda.empty_cache()
                del system
                del trainer

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
