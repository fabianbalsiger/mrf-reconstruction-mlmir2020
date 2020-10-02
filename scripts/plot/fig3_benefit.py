import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import mrf.data.definition as defs
import mrf.plot.labeling as pltlbl


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # fig3
    prepare_and_plot(
        path=os.path.join(out_dir, 'fig3.pdf'),
        legend_title='Relative error $INN_{bwd}$ - INN (%)',
        invfwdbwd_result_dir='./out/model-dir_invfwdbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/0.0e+00',
        comparison_result_dir='./out/model-dir_invbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/0.0e+00'
    )

    # supplementary figure 2
    prepare_and_plot(
        path=os.path.join(out_dir, 'sfig2.pdf'),
        legend_title='Relative error Cohen et al. - INN (%)',
        invfwdbwd_result_dir='./out/model-dir_invfwdbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/0.0e+00',
        comparison_result_dir='./out/model-dir_cohen_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/0.0e+00'
    )


def prepare_and_plot(path, legend_title, invfwdbwd_result_dir, comparison_result_dir):

    invfwdbwd_mr_params_pred, mr_params_ref = load_data(invfwdbwd_result_dir)
    comp_mr_params_pred, comp_mr_params_ref = load_data(comparison_result_dir)

    if (mr_params_ref != comp_mr_params_ref).all():
        raise ValueError('reference parameter are different')

    ff_unique = np.unique(mr_params_ref[:, defs.MR_PARAMS.index(defs.ID_MAP_FF)])
    ff_unique = ff_unique[::-1]
    t1h2o_unique = np.unique(mr_params_ref[:, defs.MR_PARAMS.index(defs.ID_MAP_T1H2O)])


    t1h2o_err_diff_map = get_error_map(invfwdbwd_mr_params_pred, comp_mr_params_pred, mr_params_ref,
                                       defs.ID_MAP_T1H2O, ff_unique, t1h2o_unique)
    t1fat_err_diff_map = get_error_map(invfwdbwd_mr_params_pred, comp_mr_params_pred, mr_params_ref,
                                       defs.ID_MAP_T1FAT, ff_unique, t1h2o_unique)

    plot_and_save_figure(path, t1h2o_err_diff_map, t1fat_err_diff_map, ff_unique, t1h2o_unique, legend_title)


def plot_and_save_figure(path, t1h2o_err_diff_map, t1fat_err_diff_map, ff_unique, t1h2o_unique, legend_title: str):
    y_axis_ticks = [f'{n:.2f}' for n in ff_unique]
    x_axis_ticks = [f'{n:d}' for n in t1h2o_unique.astype(dtype=np.int)]

    y_label = f'{pltlbl.get_map_description(defs.ID_MAP_FF, True)}'
    x_label = f'{pltlbl.get_map_description(defs.ID_MAP_T1H2O, True)}'

    with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        title = f'{pltlbl.get_map_description(defs.ID_MAP_T1H2O, False)}'
        add_rel_error_heatmap(ax1, t1h2o_err_diff_map, x_axis_ticks, y_axis_ticks,
                              title, legend_title, x_label, y_label, cb_label_pad=10)

        title = f'{pltlbl.get_map_description(defs.ID_MAP_T1FAT, False)}'
        add_rel_error_heatmap(ax2, t1fat_err_diff_map, x_axis_ticks, y_axis_ticks,
                              title, legend_title, x_label, y_label, cb_label_pad=10)

        fig.tight_layout(w_pad=5)
        plt.savefig(path)
        plt.close()


def add_rel_error_heatmap(ax, data: np.ndarray, x_axis_ticks: list, y_axis_ticks: list,
                          title: str, cbar_title: str, x_label: str, y_label: str,
                          invert_cbar: bool = False, cb_label_pad=0):
    """data.shape = (y,x)-axis"""
    cmap = 'viridis' if not invert_cbar else 'viridis_r'
    heatmap = ax.imshow(data, cmap=cmap, aspect=0.65)

    ax.set_xticks(np.linspace(0, data.shape[1] - 1, len(x_axis_ticks)))
    ax.set_yticks(np.linspace(0, data.shape[0] - 1, len(y_axis_ticks)))
    ax.set_xticklabels(x_axis_ticks, rotation=45)
    ax.set_yticklabels(y_axis_ticks)

    if title:
        ax.set_title(title, fontweight='bold', fontsize=18, pad=15)
    ax.set_xlabel(x_label, fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=14, labelpad=10)

    # create colorbar
    # see https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph for fraction
    # note the aspect above
    cbar = ax.figure.colorbar(heatmap, ax=ax, fraction=0.036, pad=0.01)
    cbar.ax.set_ylabel(cbar_title, fontweight='bold', fontsize=14, labelpad=cb_label_pad)


def load_data(result_dir):
    mr_params_pred = np.load(os.path.join(result_dir, 'mr_parameters_pred.npy'))
    mr_params_ref = np.load(os.path.join(result_dir, 'mr_parameters_ref.npy'))
    return mr_params_pred, mr_params_ref


def get_rel_abs_error(mr_params_pred, mr_params_ref, eps: float = 1e-8):
    rel_error = np.abs((mr_params_pred - mr_params_ref + eps) / (mr_params_ref + eps))
    return rel_error * 100


def get_error_map(our_mr_params_pred, baseline_mr_params_pred, mr_params_ref, mr_param, ff_unique, t1h2o_unique):

    param_idx = defs.MR_PARAMS.index(mr_param)
    our_rel_error = get_rel_abs_error(our_mr_params_pred[:, param_idx], mr_params_ref[:, param_idx])
    baseline_rel_error = get_rel_abs_error(baseline_mr_params_pred[:, param_idx], mr_params_ref[:, param_idx])

    rel_error_diff = baseline_rel_error - our_rel_error

    error_diff_map = np.zeros((ff_unique.size, t1h2o_unique.size))
    sum_map = np.zeros((ff_unique.size, t1h2o_unique.size))

    for idx, (ff, t1h2o, _, _, _) in enumerate(mr_params_ref):
        ff_idx = np.where(ff_unique == ff)
        t1h2o_idx = np.where(t1h2o_unique == t1h2o)
        error_diff_map[ff_idx, t1h2o_idx] += rel_error_diff[idx]
        sum_map[ff_idx, t1h2o_idx] += 1

    error_diff_map /= sum_map
    return error_diff_map


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Plot SNR experiments.')

    parser.add_argument(
        '--out_dir',
        type=str,
        default='./fig3',
        help='Path to the output directory.'
    )

    args = parser.parse_args()
    main(args.out_dir)
