import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import mrf.plot.labeling as pltlbl


def bland_altman_plot(path, data1: np.ndarray, data2: np.ndarray, variable_name, fontsize: float = 12):
    with plt.rc_context({'font.weight': 'bold', 'font.size': fontsize, 'mathtext.default': 'regular'}):
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2
        md = np.mean(diff)  # mean of the difference
        sd = np.std(diff, axis=0)  # standard deviation of the difference

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
        ax.scatter(mean, diff, s=2, color='black')

        # ax.set_title('Bland-Altman')
        ax.set_ylabel('$\Delta${}'.format(variable_name), fontweight='bold', fontsize=fontsize)
        ax.set_xlabel(variable_name, fontweight='bold', fontsize=fontsize)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.axhline(md, color='gray', linestyle='-')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

        # https://stackoverflow.com/questions/43675355/python-text-to-second-y-axis?noredirect=1&lq=1
        _, x = plt.gca().get_xlim()
        plt.text(x, md + 1.96 * sd, '+1.96 SD', ha='left', va='bottom')
        plt.text(x, md + 1.96 * sd, '{:.3f}'.format(md + 1.96 * sd), ha='left', va='top')

        plt.text(x, md - 1.96 * sd, '-1.96 SD', ha='left', va='bottom')
        plt.text(x, md - 1.96 * sd, '{:.3f}'.format(md - 1.96 * sd), ha='left', va='top')

        plt.text(x, md, 'Mean', ha='left', va='bottom')
        plt.text(x, md, '{:.3f}'.format(md), ha='left', va='top')

        fig.subplots_adjust(right=0.89)  # adjust slightly such that "+1.96 SD" is not cut off

        plt.savefig(path)
        plt.close()


def residual_plot(path, predicted, reference, x_label, y_label, fontsize: float = 12):
    with plt.rc_context({'font.weight': 'bold', 'font.size': fontsize, 'mathtext.default': 'regular'}):
        # Create the plot object
        _, ax = plt.subplots(figsize=(8, 6))

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        residuals = reference - predicted

        ax.scatter(predicted, residuals, s=2, color='black')

        ax.set_xlabel(x_label, fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=fontsize)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.savefig(path)
        plt.close()


def scatter_plot(path, x_data, y_data, x_label, y_label,
                 with_regression_line: bool = True, with_abline: bool = True, fontsize: float = 12):
    with plt.rc_context({'font.weight': 'bold', 'font.size': fontsize, 'mathtext.default': 'regular'}):
        _, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(x_data, y_data, s=2, color='black')

        if with_abline:
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x, color='gray')

        if with_regression_line:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            fit = p(x_data)

            # get the coordinates for the fit curve
            c_y = [np.min(fit), np.max(fit)]
            c_x = [np.min(x_data), np.max(x_data)]

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_data, y_data)
            regression = 'y = {:.3f}x {} {:.3f}'.format(slope, '+' if intercept > 0 else '-', abs(intercept))
            correlation = 'r = {:.3f}, {}'.format(r_value, pltlbl.get_p_value(p_value))

            # plot line of best fit
            ax.plot(c_x, c_y, color='gray', linestyle='dashed', label=regression)
            ax.plot([], [], ' ', label=correlation)  # "plot" to show r in legend

            # for legend placement see: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
            ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))

        ax.set_xlabel(x_label, fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=fontsize)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    """Testing routine for the plots"""
    import os
    import mrf.data.definition as defs

    root_dir = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(42)

    unit = pltlbl.get_map_description(defs.ID_MAP_FF, True)

    # plot a synthetic Bland-Altman plot
    data_ref = np.random.rand(200)
    data_pred = np.random.rand(200)
    bland_altman_plot(os.path.join(root_dir, 'bland-altman.png'), data_pred, data_ref, unit)

    # plot a synthetic correlation plot
    scatter_plot(os.path.join(root_dir, 'scatter.png'), data_ref, data_pred,
                 f'Reference {unit}', f'Predicted {unit}', with_regression_line=True, with_abline=True)

    # plot a synthetic residual plot
    residual_plot(os.path.join(root_dir, 'residual.png'), data_pred, data_ref,
                 f'Predicted {unit}', f'Residual {unit}')
