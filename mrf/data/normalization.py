import numpy as np

import mrf.data.definition as defs


def de_normalize(data: np.ndarray, minmax_tuple: tuple):
    return data * (minmax_tuple[1] - minmax_tuple[0]) + minmax_tuple[0]


def de_normalize_mr_parameters(data: np.ndarray, mr_param_ranges,
                               mr_params=defs.MR_PARAMS):
    data_de_normalized = data.copy()
    for idx, mr_param in enumerate(mr_params):
        data_de_normalized[:, idx] = de_normalize(data[:, idx], mr_param_ranges[mr_param])
    return data_de_normalized


def normalize(data: np.ndarray, minmax_tuple: tuple):
    return (data - minmax_tuple[0])/(minmax_tuple[1] - minmax_tuple[0])


def normalize_fingerprint(fingerprint: np.ndarray):
    return fingerprint / np.linalg.norm(fingerprint, 2)
