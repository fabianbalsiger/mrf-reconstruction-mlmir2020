import os

import numpy as np
import torch.utils.data as data

import mrf.data.definition as defs
import mrf.data.normalization as norm
import mrf.data.pickle as pkl


class NumpyMRFDataset(data.Dataset):

    def __init__(self, dataset_dir: str, index_selection: list = None, transform=None) -> None:
        super().__init__()
        self.mr_params = np.load(os.path.join(dataset_dir, defs.FILE_NAME_PARAMETERS), mmap_mode='r')
        self.fingerprints = np.load(os.path.join(dataset_dir, defs.FILE_NAME_FINGERPRINTS), mmap_mode='r')
        mins = pkl.load(os.path.join(dataset_dir, defs.FILE_NAME_PARAMETERS_MIN))
        maxs = pkl.load(os.path.join(dataset_dir, defs.FILE_NAME_PARAMETERS_MAX))
        self.mr_param_ranges = {k: (mins[k], maxs[k]) for k in mins}

        if index_selection is not None:
            indexes = np.asarray([int(k) for k in index_selection])
            indexes.sort()
        else:
            indexes = np.arange(self.mr_params.shape[0])
        self.indexes = indexes
        self.transform = transform

    def __len__(self) -> int:
        return self.indexes.shape[0]

    def __getitem__(self, index: int):
        sample = {defs.KEY_MR_PARAMS: np.asarray(self.mr_params[self.indexes[index]]),
                  defs.KEY_FINGERPRINTS: np.asarray(self.fingerprints[self.indexes[index]])}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_mr_param_range(self, mr_param: str):
        if mr_param not in self.mr_param_ranges:
            raise ValueError(f'Param "{mr_param}" unknown')
        return self.mr_param_ranges[mr_param]
