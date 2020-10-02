import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mrf.data.definition as defs
import mrf.plot.labeling as pltlbl


def plot_snr(path: str,
             means, stds, x_ticks, legend_labels,
             title: str, x_label: str, y_label: str,
             y_limits = None,
             fontsize: float = 9.5,
             linewidth: float = 2.5,
             errorbar_linewidth: float = 1.5,
             errorbarwidth: float = 1.0,
             legend_right: bool = True):
    with plt.rc_context({'font.weight': 'bold', 'font.size': fontsize, 'mathtext.default': 'regular'}):
        fig, ax = plt.subplots()

        n = len(legend_labels)
        colors = cm.get_cmap('viridis')(np.linspace(0, 1, n))

        total_errorbar_width = errorbarwidth * n if n <= 2 else errorbarwidth * (n - 1)
        positions = np.linspace(-total_errorbar_width / 2 + errorbarwidth / 2,
                                total_errorbar_width / 2 - errorbarwidth / 2, n)

        for idx, (color, position_offset) in enumerate(zip(colors, positions)):
            ax.errorbar(x_ticks + position_offset, means[idx],
                        yerr=stds[idx],
                        fmt='-',
                        color=color,
                        label=legend_labels[idx],
                        linewidth=linewidth,
                        elinewidth=errorbar_linewidth)

        ax.set_xticks(x_ticks)
        if y_limits:
            ax.set_ylim(y_limits[0], y_limits[1])

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(x_label, fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=fontsize)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # for legend placement see: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        if legend_right:
            ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(frameon=False, loc='upper center', ncol=n, bbox_to_anchor=(0.5, 1.2))

        fig.tight_layout()
        plt.savefig(path)
        plt.close()


def load_csv(path, mr_param, snrs):
    mr_param = defs.trim_param(mr_param)

    df = pd.read_csv(path, delimiter=';')
    df = df[df['MAP'] == mr_param]  # filter MR param

    means = []
    stds = []
    for snr in snrs:
        means.append(df[(df['METRIC'] == 'REL_ERR') & (df['SNR'] == snr)]['MEAN'].values[0] * 100)
        stds.append(df[(df['METRIC'] == 'REL_ERR') & (df['SNR'] == snr)]['STD'].values[0] * 100)

    return means, stds


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    methods = get_method_to_path()

    snrs = [5, 10, 15, 20, 25, 30, 35, 40]  # in dB
    method_to_readable = {
        'invfwdbwd': 'INN',
        'invbwd': '$INN_{bwd}$',
        'cohen': 'Cohen et al.',
        # 'oksuz': 'Oksuz et al.',
        # 'song': 'Song et al.',
        # 'hoppe': 'Hoppe et al.'
    }

    fontsize = 11.5
    file_format = '.png'

    for idx, mr_param in enumerate(defs.MR_PARAMS):
        means = []
        stds = []
        method_names = []
        for method, csv_file in methods.items():
            if method not in method_to_readable:
                continue
            method_names.append(method_to_readable[method])
            means_, stds_ = load_csv(csv_file, mr_param, snrs)
            means.append(means_)
            stds.append(stds_)

        plot_snr(os.path.join(out_dir, defs.trim_param(mr_param) + file_format),
                 means, stds, snrs, method_names,
                 pltlbl.get_map_description(mr_param, False),
                 'Signal-to-noise ratio (dB)',
                 'Relative error (%)',
                 fontsize=fontsize,
                 linewidth=2., errorbar_linewidth=1., errorbarwidth=1.0,
                 legend_right=True)


def get_method_to_path():
    method_to_path = {
        'invfwdbwd': f'./out/model-dir_invfwdbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/snr/snr_results.csv',
        'invbwd': f'./out/model-dir_invbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/snr/snr_results.csv',
        'cohen': f'./out/model-dir_cohen_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/snr/snr_results.csv',
        'hoppe': f'./out/model-dir_hoppe_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/snr/snr_results.csv',
        'oksuz': f'./out/model-dir_oksuz_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/snr/snr_results.csv',
        'song': f'./out/model-dir_song_lr=1e-3_bs=200_y-noise=3e-2/test/test-dir_test_basic/snr/snr_results.csv',
    }
    return method_to_path


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Plot SNR experiments.')

    parser.add_argument(
        '--out_dir',
        type=str,
        default='./fig2',
        help='Path to the output directory.'
    )

    args = parser.parse_args()
    main(args.out_dir)
