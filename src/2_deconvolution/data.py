import os

import h5py
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm
import pandas as pd
import copy
import torch
from torch.utils.data import DataLoader, Dataset
import random

SAMPLE_ID_TEST = "13_1226_SMTG"
SAMPLE_ID_VAL = "13_0038_SMTG"
MIXTUREFIX = "_100-800__pseudobulk_data.parquet.gzip"
MIXTUREFIX = "_pseudobulk_data.parquet.gzip"
SEPARATEFIX = "_100-800__celltype_specific.npz"
SEPARATEFIX = "_celltype_specific.npz"
class SeparationDataset(Dataset):
    def __init__(self,  mixtures,
            separate_signal, celltypes, hdf_dir, partition,
            in_memory=False,
            gene_filtering=None,
            data_transform=None, binarize=False, normalize=False,
            binarize_input=False, ratio=False, offset=1, level=None,
            force_rewriting=False, normalize_peaks=False,
            normalizeMax=False,
            pure=False, cut=False,
            #logtransform=False,
            use_only_all_cells_mixtures=False,
            mean_expression=None, celltype_to_use=None, **kwargs):
        '''
        Initialises a source separation dataset
        :param data: HDF cell data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the cell for each example (subsampling the cell)
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        self.cut = cut
        self.normalizeMax = normalizeMax
        self.logtransform = logtransform
        self.level = level
        self.gene_filtering = gene_filtering
        os.makedirs(hdf_dir, exist_ok=True)
        self.celltypes = copy.deepcopy(celltypes)

        if self.level is not None:
            self.hdf_dir = os.path.join(hdf_dir,
                            partition + str(self.level)+ ".hdf5")
        elif use_only_all_cells_mixtures:
            self.hdf_dir = os.path.join(hdf_dir,
                                    partition
                                    + str(self.level)
                                    + "only_mixtures" +
                                    ".hdf5")
        elif pure:
            self.hdf_dir = os.path.join(hdf_dir,
                                    partition
                                    + str(self.level)
                                    + "pure" +
                                    ".hdf5")

        else:
            self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
            self.level = len(self.celltypes)
        print(self.hdf_dir)
        self.celltypes_to_use = celltype_to_use
        self.in_memory = in_memory
        if not self.gene_filtering is None:
                mask = pd.read_csv(self.gene_filtering)
                mask.set_index("celltype", inplace=True)
                self.mask = mask

        if not os.path.exists(self.hdf_dir) or force_rewriting:
            self.mixtures = mixtures
            self.cell_transform = data_transform
            (self.celltype,
             separate_signal,
             ) = gatherCelltypes(celltype_to_use,
                                            separate_signal, self.celltypes)

            print(self.celltypes_to_use)
            if binarize:
                separate_signal = binarizeSeparateSignal(separate_signal,
                                                    self.celltypes)
            if binarize_input:
                self.mixtures.iloc[:,:-1] = binarizeSignal(mixtures.iloc[:,:-1])

            if normalize_peaks:
                self.mixtures.iloc[:,:-1] = normalizePeaks(mixtures.iloc[:,:-1],
                                                            mean_expression)
                separate_signal = normalizePeaks_signal(separate_signal,
                                                            mean_expression)

            if normalize:
                #separate_signal = normalizeSignal(separate_signal)
                self.mixtures.iloc[:,:-1]  = normalizeMixture(self.mixtures.iloc[:,:-1] )

            if ratio:
                separate_signal = ratioSignal(separate_signal,
                                              mixtures.iloc[:,:-1].values,
                                                offset=offset)
            if not self.gene_filtering is None:
                separate_signal = separate_signal*mask.values[np.newaxis, :,:]

            # PREPARE HDF FILE

            # Check if HDF file exists already
                # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)
            self.label = mixtures["Sample_num"].values

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                #f.attrs["Sample.ID"] = self.mixtures["Sample.ID"].astype('str')
                f.attrs["instruments"] = self.celltypes
                f.attrs["celltypes"] = self.celltypes

                print("Adding atac-seq files to dataset (preprocessing)...")
                real_idx = 0
                for idx, (index, row) in enumerate(tqdm(self.mixtures.iterrows())):
                            mix_cell = np.asarray(row[:-1]).reshape((1,-1)).astype("float32")
                            source_cells = []
                            #for j, source in enumerate(celltypes):
                            #    # In this case, read in cell and convert to target sampling rate
                            #    source_cell =  separate_signal[idx, j, : ]
                            #    source_cells.append(source_cell)
                            source_cells =  separate_signal[idx, :, : ]# np.stack(source_cells, axis=0)
                            assert(source_cells.shape[1] == mix_cell.shape[1])

                            # Add to HDF5 file
                            grp = f.create_group(str(real_idx))
                            grp.create_dataset("inputs", shape=mix_cell.shape,
                                    dtype=mix_cell.dtype, data=mix_cell)
                            grp.create_dataset("targets",
                                        shape=source_cells.shape,
                                        dtype=source_cells.dtype,
                                        data=source_cells)
                            lab = [row[-1].encode("ascii", "ignore")]
                            grp.create_dataset("label",
                                    shape=len(lab),
                                    dtype="S10",
                                        data=lab)
                            grp.attrs["length"] = mix_cell.shape[1]
                            grp.attrs["target_length"] = source_cells.shape[1]
                            real_idx +=1


        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            nb_sample = len(f)
            #lengths = [f[str(song_idx)].attrs[
            #        "target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
           # lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        #self.start_pos = SortedList(np.cumsum(lengths))
        #print("start pos : " + str(self.start_pos))
        self.length = nb_sample
        self.partition = partition
        print("lenght : " + str(self.length))

        #self.length = separate_signal.shape[0]

    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        cell = self.hdf_dataset[str(index)]["inputs"][:,:].astype("float32")
        targets = self.hdf_dataset[str(index)]["targets"][:,:].astype("float32")
        if self.cut:
            targets[targets<=3] = 0
        #if self.celltypes_to_use is not None:
        #    targets = targets[self.celltypes_to_use,:]
        if self.normalizeMax:
            cell, targets = normalizeMaxPeak(cell, targets)
        #if self.logtransform:
        #    print("LOG TRANSFORM")
        #    cell, targets = logtransform(cell, targets)
        mix = torch.from_numpy(cell.squeeze())
        ilens = mix.shape[0]
        ref = torch.from_numpy(targets)
        if self.partition == "test":
            label = self.hdf_dataset[str(index)]["label"][0].astype(str)
            return [mix, ref, label]


        return [mix, ref]

    def __len__(self):
        return self.length


def binarizeSeparateSignal(separate_signal, celltypes):
    new_signals = np.zeros_like(separate_signal)
    for k, cell in enumerate(celltypes):
        tmp = separate_signal[:,k,:]
        tmp[tmp != 0] = 1
        #tmp = np.where((tmp.T > tmp.mean(1).T).T & (tmp>tmp.mean(0)), 1,0)
        new_signals[:,k,:] = tmp
    return new_signals


def normalizeSignal(separate_signal):
    normalizeSignal = np.zeros_like(separate_signal)
    for k in range(separate_signal.shape[1]):
        for i in range(separate_signal.shape[0]):
            max_s = separate_signal[i,k,:].max()
            min_s = separate_signal[i,k,:].min()
            if max_s != 0:
                normalizeSignal[i,k,:] = (separate_signal[i,k,:] - min_s)/(max_s-min_s)
    return normalizeSignal

def normalizeMixture(mixture):
    normalizeSignal = np.zeros_like(mixture.values)
    for i in range(mixture.shape[0]):
        max_s = mixture.iloc[i,:].max()
        min_s = mixture.iloc[i,:].min()
        if max_s != 0:
            normalizeSignal[i,:] = (mixture.iloc[i,:] - min_s)/(max_s-min_s)
    return normalizeSignal


def binarizeSignal(mixtures):
    new_signals = np.zeros_like(mixtures.values)
    new_signals[mixtures.values > 0] = 1
    return new_signals

def ratioSignal(separate_signal, mixtures, offset=1):
    ratioSignal = np.zeros_like(separate_signal)
    for k in range(separate_signal.shape[1]):
        for i in range(separate_signal.shape[0]):
            ratioSignal[i,k,:] = (separate_signal[i,k,:])/(mixtures[i,:] + offset)
    return ratioSignal

def normalizePeaks(mixture, mean_expression):
    return mixture - mean_expression

def normalizePeaks_signal(signals, mean_expression):
    return np.subtract(signals, mean_expression)

def normalizeMaxPeak(mixture, signals):
    max_val = np.max(mixture)
    if max_val>0:
        mixture /= max_val
        signals /= max_val
    return mixture, signals

def logtransform(mixture, signals):
    mixture = np.log(mixture)
    signals = np.log(signals)
    return mixture, signals

def filter_data(mat, annot, key="peak_type", value="Intergenic", type="mixture"):
    index_genes= np.asarray(annot[~(annot[key] ==value)].index.tolist())
    if type == "mixture":
        mixture = mat.loc[:,np.asarray(index_genes).astype(str)]
        mixture["Sample_num"] = mat["Sample_num"]
        return mixture


    elif type=="separate":
        return mat[:,:,index_genes]
    else:
        raise "NotImplemented"


def prepareData(partition,
        sample_id_test=SAMPLE_ID_TEST,
        sample_id_val=SAMPLE_ID_VAL,
        holdout=True,
        cut_val=0.2,
        binarize=False,
        name="filter_promoter_ctrl",
        dataset_dir="/dataset/",
        hdf_dir="/dataset/hdf/",
        celltypes=['AST', 'Neur', 'OPC'],
        gene_filtering=None,
        add_pure=False,
        binarize_input=False,
        normalize=False,
        normalizeMax=False,
        ratio_input=False,
        offset_input=1,
        only_training=False,
        cut=False,
        celltype_to_use=None,
        custom_testset=None,
        crop_func=None, annot=None,
        use_train=False,
        pure=False,
        limit=None,**kwargs):
    np.random.seed(seed=0)
    SP_test = sample_id_test
    if only_training:
        SP_test += "trainonly"
    if not os.path.isfile(os.path.join(hdf_dir, partition + ".hdf5")):
        mixture = pd.read_parquet(dataset_dir
                                + name
                                + MIXTUREFIX
                                )

        separate_signals = np.load(dataset_dir + name + SEPARATEFIX)["mat"]
        if only_training:
            sample_id_train = mixture["Sample_num"].unique()
            sample_id_val = sample_id_train
            sample_id_test = sample_id_train

        else:

            sample_id = mixture["Sample_num"].unique().tolist()
            sample_id_test = [it for it in sample_id if (
                                    it in sample_id_test)]
            print("sample test :" +  str(sample_id_test))
            if sample_id_val is not None:
                sample_id_val = [it for it in sample_id if (
                                        it in sample_id_val)]
            else:
                sample_id_val = []
            #assert sample_id_test in sample_id
            sample_id_train = [it for it in sample_id if it not in sample_id_test if not it in sample_id_val]
        sample_test = sample_id_test
        sample_val = sample_id_val
        sample_train = sample_id_train
        if holdout:
            sample_train = sample_train + sample_val

        if partition == "test" and not use_train:
            separate_signals = separate_signals[
                                    mixture["Sample_num"].isin(
                                                        sample_test),:,:]
            mixture = mixture[mixture["Sample_num"].isin(sample_test)]
            if limit is not None:
                idd = np.random.randint(0,len(mixture), limit)
                separate_signals = separate_signals[idd]
                mixture = mixture.iloc[idd]
        else:
            separate_signals = separate_signals[
                                                mixture["Sample_num"].isin(
                                                        sample_train),:,:]
            mixture = mixture[mixture["Sample_num"].isin(sample_train)]
        if holdout:
            n_val = int(cut_val*len(mixture))
            n_train = len(mixture)-n_val
            print("Hold out nval : %s"%str(n_val))
            print("Hold out ntrain : %s"%str(n_train))
            list_index = list(range(len(mixture)))
            random.Random(4).shuffle(list_index)
            if partition == "val":
                index_val = list_index[:n_val]
                mixture = mixture.iloc[index_val,:]
                separate_signals = separate_signals[ index_val, :,:]
            else:
                index_train = list_index[n_val:]
                mixture = mixture.iloc[index_train,:]
                separate_signals = separate_signals[index_train, :,:]
        else:
            separate_signals = separate_signals[
                                    mixture["Sample_num"].isin(
                                                        sample_val),:,:]
            mixture = mixture[mixture["Sample_num"].isin(sample_val)]


        len_train = len(mixture)
        print("len %s: "%partition + str(len(mixture)) )
        if "Unnamed: 0" in mixture.columns.tolist():
            mixture.drop("Unnamed: 0",axis=1, inplace=True)

        _data = SeparationDataset(mixture,
                               separate_signals,
                                celltypes,
                                hdf_dir,
                                partition,
                               data_transform=crop_func,
                               gene_filtering=gene_filtering,
                               binarize=binarize,
                               binarize_input=binarize_input,
                                   normalize=normalize,
                                   normalizeMax=normalizeMax,
                                   offset=offset_input,
                                    celltype_to_use=celltype_to_use,
                                    cut=cut,
                                   ratio=ratio_input)
        return _data#, annot#, test_data
    else:
        _data = SeparationDataset(None,
                                    None,
                                    celltypes,
                                    hdf_dir,
                                    partition,
                                   data_transform=crop_func,
                                   binarize=binarize,
                                   binarize_input=binarize_input,
                                   normalize=normalize,
                                   normalizeMax=normalizeMax,
                                   offset=offset_input,
                                    celltype_to_use=celltype_to_use,
                                            cut=cut,
                                           ratio=ratio_input)
        return _data



def make_dataloader(partition,
                    annot=None,
                    is_train=True,
                    data_kwargs=None,
                    num_workers=2,
                    ratio=False,
                    batch_size=16,
                    use_train=False,
                    limit=None,
                    pure=False,
                    only_training=False,
                    custom_testset=None,
                    ):
        dataset = prepareData(partition,
                **data_kwargs,
                annot=annot)
        return DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=is_train,
                        num_workers=num_workers)


def gatherCelltypes(celltype_to_use, separate_signal, celltypes):
    if celltype_to_use is not None:
        new_separate = np.zeros((separate_signal.shape[0],
                            len(celltype_to_use),
                             separate_signal.shape[2]))
        for ind,ct in enumerate(celltype_to_use):
            if ct =="Neurons" :
                get_index = [i for i,it in enumerate(celltypes) if it in ["EX","INH"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:]
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="OPCs-Oligo" :
                get_index = [i for i,it in enumerate(celltypes) if it in ["OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:]
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="MIC-OPCs-OLD":
                get_index = [i for i,it in enumerate(celltypes) if it in ["MIC", "OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:]
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="AST-OPCs-OLD":
                get_index = [i for i,it in enumerate(celltypes) if it in ["AST", "OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:]
                new_separate[:,ind,:] = tmp.copy()
            elif ct=="Glia":
                get_index = [i for i,it in enumerate(celltypes) if it in [ "AST","MIC", "OPCs","OLD"]]
                tmp = separate_signal[:,get_index[0],:].copy()
                for i in range(1, len(get_index)):
                    tmp += separate_signal[:,get_index[i],:]
                new_separate[:,ind,:] = tmp.copy()
            else:
                get_index = [i for i,it in enumerate(celltypes) if it in [ct]]
                new_separate[:,ind,:] = separate_signal[:,get_index[0],:]
    else:
            new_separate = separate_signal
    return celltype_to_use, new_separate

