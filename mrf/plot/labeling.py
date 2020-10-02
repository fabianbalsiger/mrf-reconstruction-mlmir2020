import mrf.data.definition as defs


def get_map_description(map_: str, with_unit: bool = True):
    if map_ == defs.ID_MAP_T1H2O:
        return '$\mathrm{T1_{H2O}}$ (ms)' if with_unit else '$\mathrm{T1_{H2O}}$'
    if map_ == defs.ID_MAP_T1FAT:
        return '$\mathrm{T1_{fat}}$ (ms)' if with_unit else '$\mathrm{T1_{fat}}$'
    elif map_ == defs.ID_MAP_FF:
        return 'FF'
    elif map_ == defs.ID_MAP_B0:
        return '$\Delta$f (Hz)' if with_unit else '$\Delta$f'
    elif map_ == defs.ID_MAP_B1:
        return 'B1 (a.u.)' if with_unit else 'B1'
    else:
        raise ValueError('Map {} not supported'.format(map_.replace('map', '')))


def get_p_value(p_value: float):
    if p_value < 0.001:
        return 'p < 0.001'  # ***
    else:
        return 'p = {:.3f}'.format(p_value)
