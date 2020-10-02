KEY_FINGERPRINTS = 'fingerprints'
KEY_MR_PARAMS = 'mr_params'

ID_MAP_FF = 'FFmap'
ID_MAP_T1H2O = 'T1H2Omap'
ID_MAP_T1FAT = 'T1FATmap'
ID_MAP_B0 = 'B0map'
ID_MAP_B1 = 'B1map'

MR_PARAMS = (ID_MAP_FF, ID_MAP_T1H2O, ID_MAP_T1FAT, ID_MAP_B0, ID_MAP_B1)

FILE_NAME_FINGERPRINTS = 'fingerprints.npy'
FILE_NAME_PARAMETERS = 'parameters.npy'
FILE_NAME_PARAMETERS_MIN = 'parameters_mins.pkl'
FILE_NAME_PARAMETERS_MAX = 'parameters_maxs.pkl'
FILE_NAME_PARAMETERS_UNIQUE = 'parameters_unique.pkl'


def param_idx(param: str):
    return MR_PARAMS.index(param)


def trim_param(id_:str):
    return id_.replace('map', '')
