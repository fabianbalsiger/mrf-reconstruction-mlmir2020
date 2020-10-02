import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.stats

import mrf.data.definition as defs
import mrf.plot.labeling as pltlbl


def scatter_plot(path, x_data, y_data, mask_high,
                 title: str, subtitle: str,
                 x_label, y_label,
                 low_error: tuple = None, high_error: tuple = None,
                 low_error_color='blue', high_error_color='red',
                 xlims: tuple = None):
    with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
        fig, ax = plt.subplots(figsize=(8, 6)) #[6.4 * 2, 4.8 * 2])  # double the size due to Figure arrangement

        if mask_high is None:
            ax.scatter(x_data, y_data, s=1, color='black')
        else:
            ax.scatter(x_data[~mask_high], y_data[~mask_high], s=1, color='black')
            ax.scatter(x_data[mask_high], y_data[mask_high], s=1, color='gray')
        # ax.scatter(x_data[mask_low], y_data[mask_low], s=1, color='blue')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # markers: https://matplotlib.org/3.1.3/api/markers_api.html#module-matplotlib.markers
        if low_error:
            ax.scatter(low_error[0], low_error[1], s=75, color=low_error_color, marker='v')
        if high_error:
            ax.scatter(high_error[0], high_error[1], s=75, color=high_error_color, marker='^')

        # Label the axes and provide a title
        ax.set_xlabel(x_label, fontweight='bold', fontsize=12)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=12)

        if xlims:
            ax.set_xlim(*xlims)

        if title:
            ax.set_title(title, fontweight='bold', fontsize=12)

        if subtitle:
            ax.text(0.5, 1.06, subtitle, fontsize=10, fontweight='regular',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        plt.savefig(path)
        plt.close()


def bar_plot(path, data, mr_params, title, subtitle, x_label, y_label,
             xlims: tuple = None, color='black'):
    font_size = 15.5
    with plt.rc_context({'font.weight': 'bold', 'font.size': font_size, 'mathtext.default': 'regular'}):
        fig, ax = plt.subplots()

        y_pos = np.arange(len(mr_params))
        ax.barh(y_pos, data, align='center', color=color)
        plt.yticks(y_pos, mr_params)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # label the axes and provide a title
        ax.set_xlabel(x_label, fontweight='bold', fontsize=font_size)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=font_size)

        if xlims:
            ax.set_xlim(*xlims)

        if title:
            ax.set_title(title, fontweight='bold', fontsize=font_size)

        if subtitle:
            ax.text(0.5, 1.06, subtitle, fontsize=font_size - 4, fontweight='regular',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        plt.savefig(path)
        plt.close()


class FingerprintPlottingData:

    def __init__(self, fingerprint, label, color, marker):
        self.fingerprint = fingerprint
        self.label = label
        self.color = color
        self.marker = marker


def fingerprint_comparison_plot(path: str, data: dict,
                                title: str, subtitle: str, x_label: str, y_label: str):
    font_size = 15.5
    with plt.rc_context({'font.weight': 'bold', 'font.size': font_size, 'mathtext.default': 'regular'}):
        fig, ax = plt.subplots()

        temporal_dim = data['ref'].fingerprint.size
        x_values = list(range(1, temporal_dim + 1))

        for id_, d in data.items():
            ax.errorbar(x_values, d.fingerprint, fmt=d.marker, color=d.color, label=d.label)

        ax.xaxis.set_ticks(range(0, temporal_dim + 1, 25))

        # ax.set_title(title)
        ax.set_xlabel(x_label, fontweight='bold', fontsize=font_size)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=font_size)
        ax.set_xlim([1, temporal_dim])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if title:
            ax.set_title(title, fontweight='bold', fontsize=font_size)

        if subtitle:
            ax.text(0.5, 1.06, subtitle, fontsize=font_size - 4, fontweight='regular',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


        # for legend placement see: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        # ax.legend(frameon=False, loc='upper center', ncol=len(data), bbox_to_anchor=(0.5, 1.1))
        ax.legend(frameon=False, loc='upper right')

        fig.tight_layout()
        plt.savefig(path)
        plt.close()


def shuffle(arr, random_state: int):
    prng = np.random.RandomState(random_state)
    prng.shuffle(arr)


def main(result_dir: str, out_dir: str):
    ff_threshold = 0.9
    file_format = '.pdf'

    os.makedirs(out_dir, exist_ok=True)

    y_corr = np.load(os.path.join(result_dir, 'fingerprint_correlations.npy'))
    x_hat = np.load(os.path.join(result_dir, 'mr_parameters_pred.npy'))
    x_ref = np.load(os.path.join(result_dir, 'mr_parameters_ref.npy'))
    y_hat = np.load(os.path.join(result_dir, 'fingerprint_pred.npy'))
    y_ref = np.load(os.path.join(result_dir, 'fingerprint_ref.npy'))

    # shuffle for plotting only a fraction for visualization purposes
    random_state = 43
    shuffle(y_corr, random_state)
    shuffle(x_hat, random_state)
    shuffle(x_ref, random_state)
    shuffle(y_hat, random_state)
    shuffle(y_ref, random_state)

    eps = 1e-8
    rel_error = np.abs((x_hat - x_ref + eps) / (x_ref + eps)) * 100

    # mask for t1h2o values at high FFs that are not representative
    mask_high_ff = x_ref[:, defs.MR_PARAMS.index(defs.ID_MAP_FF)] >= ff_threshold
    rel_error_for_mean = rel_error.copy()
    # rel_error_for_mean[:, defs.MR_PARAMS.index(defs.ID_MAP_T1H2O)][mask] = np.nan  # this allows to use np.nanmean()
    mean_rel_error = np.nanmean(rel_error_for_mean, axis=1)

    # use Spearman as it might not be linear
    rho, p_value = scipy.stats.spearmanr(y_corr, mean_rel_error)
    print('Spearman correlation coefficient:')
    print(' - correlation: ', rho)
    print(' - p-value:     ', p_value)

    # get only a fraction for visualization purposes
    fraction = .1
    length = int(y_corr.size * fraction)
    x_data = y_corr[:length]
    y_data = mean_rel_error[:length]
    mask_high_ff = mask_high_ff[:length]

    # get position of high and low error example
    mask_high_error = ((x_data > 0.996) & (x_data < 0.997)) & ((y_data > 15) & (y_data < 20))
    if not mask_high_error.any():
        raise ValueError('No high error example found with this setting. Adapt the value for selecting the examples')
    high_error_idx = np.nonzero(mask_high_error)[0][0]
    high_error = (x_data[high_error_idx], y_data[high_error_idx])
    mask_low_error = (x_data > 0.99935) & (y_data < 1)
    if not mask_low_error.any():
        raise ValueError('No high error example found with this setting. Adapt the value for selecting the examples')
    low_error_idx = np.nonzero(mask_low_error)[0][0]
    low_error = (x_data[low_error_idx], y_data[low_error_idx])

    high_error_color, low_error_color = cm.get_cmap('inferno')(np.linspace(0.25, 0.75, 2))
    print('Low error color:', low_error_color)
    print('High error color:', high_error_color)

    # plot
    scatter_plot(os.path.join(out_dir, 'scatter_fraction' + file_format),
                 x_data, y_data, None,
                 'Backward error vs. forward error\n\n',
                 'Spearman\'s $\\rho$: {:.3f} ({})'.format(rho, pltlbl.get_p_value(p_value)),
                 'Fingerprint inner product',
                 'Mean relative error (%)',
                 low_error=low_error, high_error=high_error,
                 low_error_color=low_error_color, high_error_color=high_error_color,
                 xlims=(0.995, 1.0))

    print('Low error example')
    print(' - Relative error:', rel_error[low_error_idx])
    print(' - MR params:', x_ref[low_error_idx])

    print('High error example')
    print(' - Relative error:', rel_error[high_error_idx])
    print(' - MR params:', x_ref[high_error_idx])

    # plot relative errors
    mr_params_readable = [pltlbl.get_map_description(mr_param, False) for mr_param in defs.MR_PARAMS][::-1]
    xlims = (0, np.max([rel_error[low_error_idx].max(), rel_error[high_error_idx].max()]))

    def plot_bar(path, idx, color):
        title = 'Estimated MR parameters $\mathbf{\hat{x}}$\n\n'
        subtitle = 'FF={:.2f}, '.format(x_hat[idx][defs.MR_PARAMS.index(defs.ID_MAP_FF)]) + \
                '$T1_{H2O}$=' + '{:.0f} ms, '.format(x_hat[idx][defs.MR_PARAMS.index(defs.ID_MAP_T1H2O)]) + \
                '$T1_{fat}$=' + '{:.0f} ms, '.format(x_hat[idx][defs.MR_PARAMS.index(defs.ID_MAP_T1FAT)]) + \
                '$\Delta$f={:.1f} Hz, B1={:.2f}'.format(x_hat[idx][defs.MR_PARAMS.index(defs.ID_MAP_B0)],
                                                        x_hat[idx][defs.MR_PARAMS.index(defs.ID_MAP_B1)])
        subtitle += '\nMean relative error between $\mathbf{x}$ and $\mathbf{\hat{x}}$: ' + '{:.1f} %'.format(np.mean(rel_error[idx]))
        bar_plot(path,
                 rel_error[idx][::-1], mr_params_readable,
                 title, subtitle, 'Relative error (%)', 'MR parameter',
                 xlims=xlims, color=color)

    plot_bar(os.path.join(out_dir, 'bar_low_error' + file_format), low_error_idx, low_error_color)
    plot_bar(os.path.join(out_dir, 'bar_high_error' + file_format), high_error_idx, high_error_color)

    # plot fingerprint comparison y and y_hat
    def plot_fingerprint(path, fingerprint_ref, fingerprint_hat, idx, color):
        title = 'Estimated fingerprint $\mathbf{\hat{y}}$\n\n'
        subtitle = 'FF={:.2f}, '.format(x_ref[idx][defs.MR_PARAMS.index(defs.ID_MAP_FF)]) + \
                '$T1_{H2O}$=' + '{:.0f} ms, '.format(x_ref[idx][defs.MR_PARAMS.index(defs.ID_MAP_T1H2O)]) + \
                '$T1_{fat}$=' + '{:.0f} ms, '.format(x_ref[idx][defs.MR_PARAMS.index(defs.ID_MAP_T1FAT)]) +\
                '$\Delta$f={:.1f} Hz, B1={:.2f}'.format(x_ref[idx][defs.MR_PARAMS.index(defs.ID_MAP_B0)],
                                                        x_ref[idx][defs.MR_PARAMS.index(defs.ID_MAP_B1)])
        subtitle += '\nInner product between $\mathbf{y}$ and $\mathbf{\hat{y}}$: ' + '{:.4f}'.format(y_corr[idx])
        fingerprint_comparison_plot(path,
                                    {'ref': FingerprintPlottingData(fingerprint_ref,
                                                                    '$\mathbf{y}$',
                                                                    'black',
                                                                    '-'),
                                     'pred': FingerprintPlottingData(fingerprint_hat,
                                                                     '$\mathbf{\hat{y}}$',
                                                                     color,
                                                                     'x')},
                                    title, subtitle,
                                    'Temporal frame', 'Signal intensity (a.u.)')

    plot_fingerprint(os.path.join(out_dir, 'fingerprint_real_low_error' + file_format),
                     y_ref[low_error_idx].real, y_hat[low_error_idx].real, low_error_idx, low_error_color)
    plot_fingerprint(os.path.join(out_dir, 'fingerprint_imag_low_error' + file_format),
                     y_ref[low_error_idx].imag, y_hat[low_error_idx].imag, low_error_idx, low_error_color)
    plot_fingerprint(os.path.join(out_dir, 'fingerprint_magnitude_low_error' + file_format),
                     np.abs(y_ref[low_error_idx]), np.abs(y_hat[low_error_idx]), low_error_idx, low_error_color)

    plot_fingerprint(os.path.join(out_dir, 'fingerprint_real_high_error' + file_format),
                     y_ref[high_error_idx].real, y_hat[high_error_idx].real, high_error_idx, high_error_color)
    plot_fingerprint(os.path.join(out_dir, 'fingerprint_imag_high_error' + file_format),
                     y_ref[high_error_idx].imag, y_hat[high_error_idx].imag, high_error_idx, high_error_color)
    plot_fingerprint(os.path.join(out_dir, 'fingerprint_magnitude_high_error' + file_format),
                     np.abs(y_ref[high_error_idx]), np.abs(y_hat[high_error_idx]), high_error_idx, high_error_color)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Plot SNR experiments.')

    parser.add_argument(
        '--result_dir',
        type=str,
        default='./out/model-dir_invfwdbwd_lr=1e-4_bs=50_y-noise=3e-2/test/test-dir_test_basic/bwdfwd/0.0e+00',
        help='Path to the result directory containing the numpy arrays with the fingerprints and MR parameters.'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='./fig4',
        help='Path to the output directory.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.out_dir)
