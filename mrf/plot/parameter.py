import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def prediction_distribution_plot(path, ref_x, pred_x, x_label):
    # with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
    _, ax = plt.subplots(1, 1, figsize=(10, 4))

    ticks = np.unique(ref_x)  # sorted
    is_integers = np.all(np.equal(np.mod(ticks, 1), 0))

    colors = cm.get_cmap('viridis')(np.linspace(0, 1, ticks.size))
    ys = np.random.normal(0.0, scale=.5, size=pred_x.shape[0])  # distribute points on y-axis

    # set true x's
    for tick, color in zip(ticks, colors):
        ax.plot((tick, tick),
                (ys.min(), ys.max()),
                color=color, label=f'{tick:.2f}' if not is_integers else f'{int(tick):d}', linewidth=.5)
    max_legend_cols = 8  # seems to work good
    ax.legend(frameon=False, loc='upper center', ncol=max_legend_cols, bbox_to_anchor=(0.5, 1.18))

    ax.scatter(pred_x, ys, c=ref_x, cmap='viridis', s=1., alpha=0.5)

    ax.set_xticks(ticks)
    if is_integers:
        ax.set_xticklabels([f'{int(tick):d}' for tick in ticks])
    else:
        ax.set_xticklabels([f'{tick:.2f}' for tick in ticks])
    ax.set_yticks([])
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel(x_label)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
