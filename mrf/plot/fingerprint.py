import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plt_correlation(path: str, data: np.ndarray, x_axis_ticks: list, y_axis_ticks: list,
                 title: str, cbar_title: str, x_label: str, y_label: str,
                 invert_cbar: bool = False, fontsize: float = 12):
    """data.shape = (y,x)-axis"""
    with plt.rc_context({'font.weight': 'bold', 'font.size': fontsize, 'mathtext.default': 'regular'}):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        cmap = 'viridis' if not invert_cbar else 'viridis_r'
        heatmap = ax.imshow(data, cmap=cmap, aspect=0.75)

        ax.set_xticks(np.linspace(0, data.shape[1] - 1, len(x_axis_ticks)))
        ax.set_yticks(np.linspace(0, data.shape[0] - 1, len(y_axis_ticks)))
        ax.set_xticklabels(x_axis_ticks, rotation=45)
        ax.set_yticklabels(y_axis_ticks)

        if title:
            ax.set_title(title, fontweight='bold', fontsize=fontsize)
        ax.set_xlabel(x_label, fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=fontsize)

        # create colorbar
        # see https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph for fraction
        # note the aspect above
        cbar = ax.figure.colorbar(heatmap, ax=ax, fraction=0.048*data.shape[0]/data.shape[1], pad=0.01)
        cbar.ax.set_ylabel(cbar_title, fontweight='bold', fontsize=fontsize)

        fig.tight_layout()
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    """Testing routine for the plots"""
    import os
    import mrf.data.definition as defs
    import mrf.plot.labeling as pltlbl

    root_dir = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(42)

    # plot a synthetic correlation matrix
    # assume y-axis is fat fraction and x-axis is water T1
    correlations = np.random.rand(200, 100)
    plt_correlation(os.path.join(root_dir, 'correlation.png'), correlations,
                    x_axis_ticks=[f'{n:.1f}' for n in np.arange(0, 1.1, 0.1)],
                    y_axis_ticks=[f'{n:d}' for n in np.arange(500, 2350, 100)],
                    title='Test Plot', cbar_title='Correlation',
                    x_label=pltlbl.get_map_description(defs.ID_MAP_FF, True),
                    y_label=pltlbl.get_map_description(defs.ID_MAP_T1H2O, True),
                    invert_cbar=False)
