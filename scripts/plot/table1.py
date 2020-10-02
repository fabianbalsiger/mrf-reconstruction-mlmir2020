import os

import scipy.stats as stats
import numpy as np

import mrf.data.definition as defs
import mrf.evaluation.metric as metric


def main():

    noise_level = 0.0

    params_ref = None
    results = {}
    for method, path in get_method_to_path(noise_level).items():
        ref, predicted = load_data(path)
        if params_ref is None:
            params_ref = ref
        assert np.allclose(ref, params_ref)

        ae = calculate_ae(predicted, ref)
        re = calculate_re(predicted, ref)
        r2 = np.zeros(predicted.shape[-1], dtype=np.float)
        for param_idx, mr_param in enumerate(maps):
            r2[param_idx] = calucalate_r2(predicted[:, param_idx], ref[:, param_idx])

        for param_idx, mr_param in enumerate(maps):
            results.setdefault('R2', {}).setdefault(defs.trim_param(mr_param), {})[method] = r2[param_idx]
            results.setdefault('ABS_ERR', {}).setdefault(defs.trim_param(mr_param), {})[method] = ae[:, param_idx]
            results.setdefault('REL_ERR', {}).setdefault(defs.trim_param(mr_param), {})[method] = re[:, param_idx]


    # ae_statistics(results)

    print_table(results)


def calucalate_r2(predicted, ref):
    # from pymia CoefficientOfDetermination
    # todo: take the actual function out of Metric to be able to call it also from outside
    ref = ref.flatten()
    predicted = predicted.flatten()

    sse = sum((ref - predicted) ** 2)
    tse = (len(ref) - 1) * np.var(ref, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score


def calculate_ae(predicted, ref):
    return np.abs(ref - predicted)


def calculate_re(predicted, ref, percent=True):
    factor = 100.0 if percent else 1.0
    return metric.relative_error(predicted, ref) * factor


def ae_statistics(results):
    map_results = results['ABS_ERR']
    for i, map_ in enumerate(maps):
        print(f'\n\n{defs.trim_param(map_)}')
        map_ae = map_results[defs.trim_param(map_)]

        # r = {}
        # for method, arr in map_ae.items():
        #     r[method] = stats.shapiro(arr)
        # print('Normality shapiro')
        # print('\t'.join(f'{k}: W={w}, p={p}' for k, (w, p) in r.items()))
        # print(f'reject normality for any: {any([p < 0.05 for (_, p) in r.values()])}')
        #
        # s, p = stats.kruskal(*(arr for arr in map_ae.values()))
        # print('\nKruskal')
        # print(f'{map_} -> p-val: {p}')
        # print(f'any difference in methods {p < 0.05}')

        maes = [(m, arr.mean()) for m, arr in map_ae.items()]
        sorted_maes = sorted(maes, key=lambda tup: tup[1])

        best_method, _ = sorted_maes[0]
        r = {}
        for method, _ in sorted_maes[1:]:
            r[method] = stats.wilcoxon(map_ae[best_method], map_ae[method])
        print('Wilcoxon')
        print('\t'.join(f'{best_method}-{k}: s={s}, p={p}' for k, (s, p) in r.items()))


def get_err_lines(results, error_entry, metric_title, precisions, with_units, percentage_units=False):
    std_scale = 0.75
    unit_scale = 0.75
    precision_mean = precisions
    precision_std = precisions

    map_results = results[error_entry]

    lines = []
    for i, map_ in enumerate(maps):
        map_err = map_results[defs.trim_param(map_)]

        # calculate mean absolute error (and std) and find the best methods
        mean = {}
        std_err = {}
        min_mean_err = np.inf
        for method, arr in map_err.items():
            val = arr.mean().round(precision_mean[i])
            if val < min_mean_err:
                min_mean_err = val
            mean[method] = val
            std_err[method] = arr.std().round(precision_std[i])
        min_methods = [m for m, v in mean.items() if v == min_mean_err]

        descr = get_map_description(map_, with_unit=with_units, with_percentage=percentage_units, scale=unit_scale)
        map_result = [metric_title if i == 0 else '',  descr]
        for method in method_to_str:
            m = f'{mean[method]:.{precision_mean[i]}f}'
            if method in min_methods:
                m = f'\\textbf{{{m}}}'
            m += _scale(f'$\,\pm\,${std_err[method]:.{precision_std[i]}f}', scale=std_scale)
            # m += f'~({method_mae["STD"].item():.{precision_std[i]}f})'
            map_result.append(m)
        line = ' & '.join(map_result)
        lines.append(line)
    return lines


def _scale(s, scale:float = None):
    return f'\scalebox{{{scale}}}{{{s}}}' if scale is not None else s


def get_r2_lines(results):
    precision = 3
    r2_results = results['R2']

    lines = []
    for i, map_ in enumerate(maps):
        map_r2 = r2_results[defs.trim_param(map_)]

        r2 = {}
        max_r2 = -np.inf
        for method, val in map_r2.items():
            val_p = val.round(precision)
            if val_p > max_r2:
                max_r2 = val_p
            r2[method] = val_p

        max_methods = [m for m, v in r2.items() if v == max_r2]

        map_result = ['R\\textsuperscript{2}' if i == 0 else '', get_map_description(map_, with_unit=False)]
        for method in method_to_str:
            m = f'{map_r2[method]:.{precision}f}'
            if method in max_methods:
                m = f'\\textbf{{{m}}}'
            map_result.append(m)
        line = ' & '.join(map_result)
        lines.append(line)
    return lines


def get_header():
    m = ['Metric', 'MR parameter'] + [method_to_str[k] for k in method_to_str]
    return ' & '.join(m)


def print_table(results):
    nb_methods = len(method_to_str)
    offset = '\t'

    lines = [
        '\\toprule',
        f'& & \multicolumn{{{nb_methods}}}{{c}}{{Method}} \\\\',
        f'\cmidrule{{3-{2 + nb_methods}}}',
        get_header() + '\\\\',
        '\midrule',
        *[line + ' \\\\' for line in get_err_lines(results, 'ABS_ERR', 'MAE', (3, 1, 1, 3, 3), True)],
        '\midrule',
        *[line + ' \\\\' for line in get_err_lines(results, 'REL_ERR', 'MRE', (2, 2, 2, 2, 2), False, True)],
        '\midrule',
        *[line + ' \\\\' for line in get_r2_lines(results)],
        '\\bottomrule'
    ]

    lines = [f'{offset}\t' + line for line in lines]
    desc = 'll@{\hspace{0.5em}}' + '@{\hspace{1em}}'.join(['c']*nb_methods)
    lines.insert(0, f'{offset}\\begin{{tabular}}{{{desc}}}')
    lines.append(f'{offset}\end{{tabular}}')

    lines_str = '\n'.join(lines)
    print(lines_str)


def get_map_description(map_, with_unit: bool = True, with_percentage = False, scale:float=None):
    if map_ == defs.ID_MAP_T1H2O:
        val = f'T1\\textsubscript{{H2O}}{_scale(" (ms)", scale)}' if with_unit else 'T1\\textsubscript{H2O}'
    elif map_ == defs.ID_MAP_T1FAT:
        val =  f'T1\\textsubscript{{fat}}{_scale(" (ms)", scale)}' if with_unit else 'T1\\textsubscript{fat}'
    elif map_ == defs.ID_MAP_FF:
        val =  'FF'
    elif map_ == defs.ID_MAP_B0:
        val =  f'$\Delta$f{_scale(" (Hz)", scale)}' if with_unit else '$\Delta$f'
    elif map_ == defs.ID_MAP_B1:
        val =  f'B1{_scale(" (a.u.)", scale)}' if with_unit else 'B1'
    else:
        raise ValueError('Map {} not supported'.format(map_.replace('map', '')))

    if with_percentage:
        val += _scale(' (\%)', scale)
    return val


maps = defs.MR_PARAMS


method_to_str = {
    'invfwdbwd': 'INN',
    'invbwd': 'INN\\textsubscript{bwd}',
    'cohen': 'Cohen et al.',
    'hoppe': 'Hoppe et al.',
    'oksuz': 'Oksuz et al.',
    'song': 'Song et al.',
}


def get_method_to_path(noise_level: float):
    noise_str = f'{noise_level:.1e}'
    method_to_path = {
        'invfwdbwd': f'./out/model-dir_invfwdbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/{noise_str}',
        'invbwd': f'./out/model-dir_invbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/{noise_str}',
        'cohen': f'./out/model-dir_cohen_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/{noise_str}',
        'hoppe': f'./out/model-dir_hoppe_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/{noise_str}',
        'oksuz': f'./out/model-dir_oksuz_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/{noise_str}',
        'song': f'./out/model-dir_song_lr=1e-3_bs=200_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/{noise_str}',
    }
    return method_to_path


def load_data(path: str):
    ref = np.load(os.path.join(path, 'mr_parameters_ref.npy'))
    pred = np.load(os.path.join(path, 'mr_parameters_pred.npy'))
    return ref, pred


if __name__ == '__main__':
    main()
