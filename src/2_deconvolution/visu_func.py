import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
PALETTE = {"AST": "#8aafa9", 
    'EXC-L23':"#ead1dc",
    'EXC-L4':"#c27ba0",
    'EXC-L5':"#741b47",
    'EXC-L6':"#d9d2e9",
    "EXC-IT":"#c27ba0",
    "EXC-L6-spe":"#d9d2e9",
    "EXC-L6_spe":"#d9d2e9",
    "EXC-L6":"#d9d2e9",
    'EXC':"#d9d2e9",
    "INH-CGE":"#f34c0d",
    "INH":"#f34c0d",
    "INH-MGE":"#FCDBCE",
    "MIC":"#165f54",
    # "ENDO-Mural":"#ab910b",
    "Endo":"#ab910b",
    "VLMC":"#fbf0ba",
    "Endo-Mural":"#ab910b",
    "OLD":"#ffc281",
    "OPC":"#7f6140"
    }
def plot_per_celltype(df_metrics_tot,
                    method,
                    savename,
                    metrics,
                    ):


    fig, axes = plt.subplots(1,2, figsize=(18,6))
    axes = axes.flatten()
    sns.set(font_scale=2, style="white")
    fontsize=18
    #tmp_method = "RandomForestRegressor"
    tmp_method = method
    df_tmp = df_metrics_tot[df_metrics_tot.method ==tmp_method]
    for indx, it in enumerate(metrics):
        tmp = df_tmp[df_tmp.metrics==it].groupby(["celltype","method", "individualID","fold"]).res.mean().reset_index()
        ax = axes[indx]
        sns.boxplot(data=tmp,x="celltype",
                hue="celltype",
                y="res", 
                palette=PALETTE,
                ax=ax,
                showmeans=True,
                dodge=False,
                   showfliers = False, 
                meanprops={"marker":"o",
                    "markerfacecolor":"black",
                    "markeredgecolor":"black",
                    "markersize":"5"},
                linewidth=0.3)#, notch=True)
        means = tmp.groupby(['celltype'])['res'].mean().round(2)
        vertical_offset = tmp['res'].mean() * 0.02 # offset from median for display
        if "mse" in it:
            ax.set_yscale("log")
        for xtick in ax.get_xticks():
            ax.text(xtick,
                    means[xtick] + vertical_offset,
                    means[xtick], 
                    horizontalalignment='center',
                    size='x-small',color='black',weight='semibold')
        ax.set_xlabel("")
        ax.set_title(tmp_method + " | " + it)
        ax.set_ylabel("")
    plt.savefig(savename + "box_comp_CV_CELLTYPE.svg",
                bbox_inches="tight")
    plt.close("all")

def plot_model_comparison(df_metrics_tot,
                        savename,
                        palette,
                          pairs,
                          show=False,
                        metrics=["spearman", "rmse"],
                        hue_order=["RandomForestRegressor", "NMF",
                            "LinearRegression", "knn"]):


    fig, axes = plt.subplots(1,len(metrics), figsize=(18,6))
    axes = axes.flatten()
    com = df_metrics_tot.method.unique()
    print(com)

    #hue_order=["Cellformer", "NMF", "KNN"]

    sns.set(font_scale=2, style="white")
    fontsize=18
    for indx, it in enumerate(metrics):
        tmp = df_metrics_tot[df_metrics_tot.metrics==it].groupby(["method","celltype", "individualID", "fold"]).res.mean().reset_index()
        ax = axes[indx]
        sns.boxplot(data=tmp,y="res",
                hue="method",
                x="method", palette=palette,
                hue_order=hue_order,
                order=hue_order,
                ax=ax,
                showfliers = False,
                showmeans=True,
                dodge=False,
                meanprops={"marker":"o",
                    "markerfacecolor":"black",
                    "markeredgecolor":"black",
                    "markersize":"5"},
                linewidth=0.8)#, notch=True)
        annotator = Annotator(ax, pairs, data=tmp,
                           y="res",
                           x="method",
                           hue="method",
                           hue_order=hue_order,
                           order=hue_order,
                                )
        annotator.configure(test='Mann-Whitney',  text_format="star", 
                           loc='inside', fontsize=8, 
                           comparisons_correction="BH")
        annotator.apply_and_annotate()
        ax.legend("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), 
                       fontsize=fontsize)
        means = tmp.groupby(['method'])['res'].mean().round(2)
        vertical_offset = tmp['res'].mean() * 0.02 # offset from median for display
        # if "mse"in it:
        #     ax.set_yscale("log")
        for xtick in ax.get_xticklabels():
            lab = xtick.get_text()
            print(lab)
            pos = xtick.get_position()[0]
            ax.text(pos,
                    means.loc[lab] + vertical_offset,
                    means.loc[lab], 
                    horizontalalignment='center',
                    size='x-small',color='black',weight='semibold')
        ax.set_xlabel("")
        if it =="pearson":
            title = "Pearson"
        elif it =="spearman":
            title = "Spearman"
        else:
            title=it
        ax.set_title(title)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=20)
        ax.legend().remove()
    plt.savefig(savename + "box_comp_CV_with_annot_test.svg",
                bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")

def plot_model_comparison_stratified_ct(df_metrics_tot,
                        savename,
                        palette,
                          pairs,
                          show=False,
                        metrics=["spearman", "rmse"],
                        hue_order=["RandomForestRegressor", "NMF",
                            "LinearRegression", "knn"]):


    fig, axes = plt.subplots(1,len(metrics), figsize=(18,6))
    axes = axes.flatten()
    com = df_metrics_tot.method.unique()
    print(com)

    #hue_order=["Cellformer", "NMF", "KNN"]

    sns.set(font_scale=2, style="white")
    fontsize=18
    for indx, it in enumerate(metrics):
        tmp = df_metrics_tot[df_metrics_tot.metrics==it].groupby(["method","celltype", "individualID", "fold"]).res.mean().reset_index()
        ax = axes[indx]
        sns.boxplot(data=tmp,y="res",
            x="celltype",
                hue="method", palette=palette,
                hue_order=hue_order,
                #order=hue_order,
                ax=ax,
                showfliers = False,
                showmeans=True,
                dodge=True,
                meanprops={"marker":"o",
                    "markerfacecolor":"black",
                    "markeredgecolor":"black",
                    "markersize":"5"},
                linewidth=0.8)#, notch=True)
        annotator = Annotator(ax, pairs, data=tmp,
                           y="res",
                           x="celltype",
                           hue="method",
                           hue_order=hue_order,
                           #order=hue_order,
                                )
        annotator.configure(test='Mann-Whitney',  text_format="star", 
                           loc='inside', fontsize="20", 
                           comparisons_correction="BH")
        annotator.apply_and_annotate()
        ax.legend("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), 
                       fontsize=fontsize)
       # means = tmp.groupby(['celltype','method'])['res'].mean().round(2)
       # vertical_offset = tmp['res'].mean() * 0.02 # offset from median for display
       # # if "mse"in it:
       # #     ax.set_yscale("log")
       # for xtick in ax.get_xticklabels():
       #     lab = xtick.get_text()
       #     print(lab)
       #     pos = xtick.get_position()[0]
       #     ax.text(pos,
       #             means.loc[lab] + vertical_offset,
       #             means.loc[lab], 
       #             horizontalalignment='center',
       #             size='x-small',color='black',weight='semibold')
        ax.set_xlabel("")
        if it =="pearson":
            title = "Pearson"
        elif it =="spearman":
            title = "Spearman"
        else:
            title=it
        ax.set_title(title)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=20)
        ax.legend().remove()
    plt.savefig(savename + "box_comp_CV_with_annot_test_stratified_Ct.svg",
                bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")
def plot_comparison_per_genes(df_metrics_tot_genes,
                            savename,
                            palette,
                            pairs,
                            metrics,
                              show=False,
                        hue_order=["RandomForestRegressor", 
                            "NMF", 
                            "LinearRegression",
                            "knn"]
                            ):

    fig, axes = plt.subplots(1,len(metrics), figsize=(18,6))
    axes = axes.flatten()
    com = df_metrics_tot_genes.method.unique()
    print(com)

    sns.set(font_scale=2, style="white")
    fontsize=18
    for indx, it in enumerate(metrics):
        tmp = df_metrics_tot_genes[df_metrics_tot_genes.metrics==it].groupby(["method","celltype","genes", "fold"]).res.mean().reset_index()
        ax = axes[indx]
        sns.boxplot(data=tmp,y="res",
                hue="method",
                x="method", palette=palette,
                hue_order=hue_order,
                order=hue_order,
                ax=ax,
                showfliers = False,
                showmeans=True,
                dodge=False,
                meanprops={"marker":"o",
                    "markerfacecolor":"black",
                    "markeredgecolor":"black",
                    "markersize":"5"},
                linewidth=0.8)#, notch=True)
        annotator = Annotator(ax, pairs, data=tmp,
                            y="res",
                            x="method",
                            hue="method",
                            hue_order=hue_order,
                            order=hue_order,
                                 )
        annotator.configure(test='Mann-Whitney',  text_format="star", 
                            loc='inside', fontsize="8", 
                            comparisons_correction="BH")
        annotator.apply_and_annotate()
        #ax.legend("")
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
        #ax.set_yticklabels(ax.get_yticklabels(), 
        #                fontsize=fontsize)
        means = tmp.groupby(['method'])['res'].mean().round(2)
        vertical_offset = tmp['res'].mean() * 0.02 # offset from median for display
        if "mse" in it:
            ax.set_yscale("log")
        for xtick in ax.get_xticklabels():
            lab = xtick.get_text()
            print(lab)
            pos = xtick.get_position()[0]
            ax.text(pos,
                    means.loc[lab] + vertical_offset,
                    means.loc[lab], 
                    horizontalalignment='center',
                    size='x-small',color='black',weight='semibold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
        ax.set_xlabel("")
        if it =="pearson":
            title = "Pearson"
        elif it =="spearman":
            title = "Spearman"
        else:
            title=it
        ax.set_title(title)
        ax.tick_params(axis="both", labelsize=20)
        ax.set_ylabel("")
    plt.savefig(savename + "box_comp_CV_per_genes.svg",
                bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")

def plot_gene_per_celltype(df_metrics_tot_genes,
                    method,
                    savename,
                    metrics,
                    ):
    fig, axes = plt.subplots(1,2, figsize=(18,6))
    axes = axes.flatten()
    sns.set(font_scale=2, style="white")
    fontsize=18
    tmp_method = method # "RandomForestRegressor"
    df_tmp = df_metrics_tot_genes[df_metrics_tot_genes.method ==tmp_method]
    for indx, it in enumerate(metrics):
        tmp = df_tmp[df_tmp.metrics==it].groupby(["method","celltype","genes", "fold"]).res.mean().reset_index()
        ax = axes[indx]
        sns.boxplot(data=tmp,x="celltype",
                hue="celltype",
                y="res", palette=PALETTE,
                ax=ax,
                showmeans=True,
                dodge=False,
                showfliers = False,
                meanprops={"marker":"o",
                    "markerfacecolor":"black",
                    "markeredgecolor":"black",
                    "markersize":"5"},
                linewidth=0.3)#, notch=True)
        means = tmp.groupby(['celltype'])['res'].mean().round(2)
        vertical_offset = tmp['res'].mean() * 0.02 # offset from median for display
        if "mse" in it:
            ax.set_yscale("log")
        for xtick in ax.get_xticks():
            ax.text(xtick,
                    means[xtick] + vertical_offset,
                    means[xtick], 
                    horizontalalignment='center',
                    size='x-small',color='black',weight='semibold')
        ax.set_xlabel("")
        ax.set_title(tmp_method + " | " + it)
        ax.set_ylabel("")
    plt.savefig(savename + "box_comp_CV_CELLTYPE_per_Gene.svg",
                bbox_inches="tight")
    plt.close("all")
