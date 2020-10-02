import os

import numpy as np

import mrf.data.definition as defs
import mrf.data.normalization as norm
import mrf.evaluation.metric as metric
import mrf.evaluation.base as evalbase
import mrf.plot.fingerprint as pltfingerprint
import mrf.plot.labeling as pltlbl
import mrf.plot.statistics as pltstat


class ForwardEvaluator(evalbase.BaseEvaluator):
    """A forward evaluator, evaluating the goodness of fingerprint predictions."""

    def __init__(self, reference, prediction, mr_params, ranges):
        """Initializes the ForwardEvaluator class.

        Args:
            reference: The reference fingerprints (normalized).
            prediction: The predicted fingerprints (normalized).
            mr_params: The normalized MR parameters the fingerprints correspond to.
            ranges: The original range of the MR parameters before normalization.
        """
        n_trs = reference.shape[1] // 2
        self.reference = reference[:, 0:n_trs] + 1j * reference[:, n_trs:]
        self.prediction = prediction[:, 0:n_trs] + 1j * prediction[:, n_trs:]
        self.mr_params = norm.de_normalize_mr_parameters(mr_params, ranges)
        self.ranges = ranges

        self.correlations = self._calculate_correlations()
        self.results = self.calculate()

    def calculate(self) -> dict:
        return {'CORR-MEAN': self.correlations.mean(), 'CORR-STD': self.correlations.std()}

    def plot(self, root_dir: str):
        ff_param_idx = defs.MR_PARAMS.index(defs.ID_MAP_FF)
        t1h2o_param_idx = defs.MR_PARAMS.index(defs.ID_MAP_T1H2O)
        ff_unique = np.unique(self.mr_params[:, ff_param_idx])
        t1h2o_unique = np.unique(self.mr_params[:, t1h2o_param_idx])

        correlation_map = np.zeros((ff_unique.size, t1h2o_unique.size))
        sum_map = np.zeros((ff_unique.size, t1h2o_unique.size))

        for idx, (ff, t1h2o, _, _, _) in enumerate(self.mr_params):
            ff_idx = np.where(ff_unique == ff)
            t1h2o_idx = np.where(t1h2o_unique == t1h2o)
            correlation_map[ff_idx, t1h2o_idx] += self.correlations[idx]
            sum_map[ff_idx, t1h2o_idx] += 1

        correlation_map /= sum_map

        ff_step = 0.1
        ff_ticks = np.arange(ff_unique.min(), ff_unique.max() + ff_step / 2, ff_step)
        t1h2o_step = 100
        t1h2o_ticks = np.arange(t1h2o_unique.min(), t1h2o_unique.max() + t1h2o_step / 2, t1h2o_step)
        pltfingerprint.plt_correlation(os.path.join(root_dir, 'correlation-ff-t1h2o.png'),
                                       correlation_map,
                                       y_axis_ticks=[f'{n:.1f}' for n in ff_ticks],
                                       x_axis_ticks=[f'{n:d}' for n in t1h2o_ticks.astype(dtype=np.int)],
                                       cbar_title='Correlation', title='',
                                       y_label=pltlbl.get_map_description(defs.ID_MAP_FF, True),
                                       x_label=pltlbl.get_map_description(defs.ID_MAP_T1H2O, True),
                                       invert_cbar=False)

    def _calculate_correlations(self) -> np.ndarray:
        correlations = np.empty((self.reference.shape[0],))
        for i, (ref_fingerprint, pred_fingerprint) in enumerate(zip(self.reference, self.prediction)):
            correlations[i] = np.abs(np.vdot(ref_fingerprint,
                                             norm.normalize_fingerprint(pred_fingerprint)))
        return correlations

    def save(self, root_dir: str):
        np.save(os.path.join(root_dir, 'correlations.npy'), self.correlations)
        np.save(os.path.join(root_dir, 'mr_params_correlations.npy'), self.mr_params)


class ForwardCorrelationEvaluator(evalbase.BaseEvaluator):
    """A correlation evaluator, evaluating the correlation of the forward predictions to the backward errors."""

    def __init__(self, reference, prediction, mr_params, mr_params_prediction, ranges, save_fingerprints=False):
        """Initializes the ForwardCorrelationEvaluator class.

        Args:
            reference: The reference fingerprints (normalized).
            prediction: The predicted fingerprints (normalized).
            mr_params: The normalized MR parameters the fingerprints correspond to.
            mr_params_prediction: The predicted MR parameters.
            ranges: The original range of the MR parameters before normalization.
            save_fingerprints: whether to save the fingerprints when save is called.
        """
        n_trs = reference.shape[1] // 2
        self.reference = reference[:, 0:n_trs] + 1j * reference[:, n_trs:]
        self.prediction = prediction[:, 0:n_trs] + 1j * prediction[:, n_trs:]
        self.mr_params = norm.de_normalize_mr_parameters(mr_params, ranges)
        self.mr_params_prediction = norm.de_normalize_mr_parameters(mr_params_prediction, ranges)
        self.ranges = ranges
        self.save_fingerprints = save_fingerprints

        self.correlations = self._calculate_correlations()
        self.relative_error = self._calculate_mr_params_relative_error()
        self.results = self.calculate()

    def calculate(self) -> dict:
        return {}

    def plot(self, root_dir: str):
        file_name = os.path.join(root_dir, 'corr-error_scatter_mean.png')
        pltstat.scatter_plot(file_name, self.correlations, self.relative_error.mean(axis=1),
                              'Inner product', 'Relative error',
                             with_abline=False, with_regression_line=False)

        for idx, mr_param in enumerate(defs.MR_PARAMS):
            param_rel_err = self.relative_error[:, idx]

            file_name = os.path.join(root_dir, f'corr-error_scatter_{defs.trim_param(mr_param)}.png')
            pltstat.scatter_plot(file_name, self.correlations, param_rel_err, 'Inner product', 'Mean relative error',
                                 with_abline=False, with_regression_line=False)

    def _calculate_correlations(self) -> np.ndarray:
        correlations = np.empty((self.reference.shape[0], ))
        for i, (ref_fingerprint, pred_fingerprint) in enumerate(zip(self.reference, self.prediction)):
            correlations[i] = np.abs(np.vdot(ref_fingerprint,
                                             norm.normalize_fingerprint(pred_fingerprint)))
        return correlations

    def _calculate_mr_params_relative_error(self):
        return metric.relative_error(self.mr_params_prediction, self.mr_params)

    def save(self, root_dir: str):
        np.save(os.path.join(root_dir, 'fingerprint_correlations.npy'), self.correlations)
        if self.save_fingerprints:
            np.save(os.path.join(root_dir, 'fingerprint_pred.npy'), self.prediction)
            np.save(os.path.join(root_dir, 'fingerprint_ref.npy'), self.reference)
