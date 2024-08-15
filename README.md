# Cellformer
An implementation of cell type-specific RNA deconvolution using Cellformer.

## Installation from source
The lastest source version of Cellformer can be accessed by running the following command:

```
git clone https://github.com/elo-nsrb/Cellformer.git
cd Cellformer
```

## Requirements

In order to install package dependencies, you will need [Anaconda](https://anaconda.org/). After installing Anaconda, please run the following command to create two conda environnements with R and Pytorch dependencies:

`.\setup.sh`

## Usage

### 1. Synthetic dataset generation
Synthetic dataset can be created from snATAC-seq gene expression matrix in [AnnData format](https://anndata.readthedocs.io/en/latest/) with `celltype` and `Sample_num` columns in `obs`.

```
python src/1-preprocessing/createSyntheticDataset.py --savepath [path_data] --filename [ anndata file] --nb_cells_per_case 2000 -nb_core 30 --name data_totnorm_lognorm_nosparse
```


### 2. Pretrained model inference and bulk deconvolution
We provided the pretrained model used in the manuscript in [Model_universal](https://github.com/elo-nsrb/CellformerRNA/tree/main/Model_universal/). The pretrained model can be used to deconvolute bulk matrix by running:

```
conda activate pytorch_env
python src/2-deconvolution/cv_inference --parent_dir Model_universal --gene_count_matrix [gene expression matrix] --type bulk --save
```

You can find an example of the expected bulk matrix format `bulk_countMatrix.txt` in [data](https://github.com/elo-nsrb/Cellformer/tree/main/data). It must have a column name 'Sample_num' and the same genes in the same order as in this [file](https://github.com/elo-nsrb/CellformerRNA/tree/main/Model_universal/input_genes.csv).

### 3. Model training / finetuning

Cellformer can be trained and vallidated from scratch using a synthetic dataset and configuration file `train.yml` (see an example in [Model_universal](https://github.com/elo-nsrb/CellformerRNA/tree/main/Model_universal/)) by running:
```
conda activate pytorch_env
python src/2-deconvolution cvTrain.py --parent_dir [model path] --model SepFormerTasNet 
```

For finetuning, please run:

```
python src/2-deconvolution cvTrain.py --parent_dir [model path] --model SepFormerTasNet --resume -resume_ckpt [pretrained model weights]
```
Please modify the path to the data folder in `train.yml`.

## Citation
- *Berson, E et al.* (2024)
- *Berson, E. et al.* (2023). **Whole genome deconvolution unveils Alzheimer’s resilient epigenetic signature**. Nature Communications, 14(1) 4947
[link](https://www.nature.com/articles/s41467-023-40611-4)



## Licence
This project is covered under the [GNU General Public License v3.0](https://github.com/elo-nsrb/Cellformer/blob/main/LICENSE)
