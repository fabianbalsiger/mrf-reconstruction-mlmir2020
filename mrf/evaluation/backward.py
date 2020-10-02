import os

import numpy as np

import mrf.data.definition as defs
import mrf.data.normalization as norm
import mrf.evaluation.base as evalbase
import mrf.evaluation.metric as metric
import mrf.plot.labeling as pltlbl
import mrf.plot.parameter as pltparam
import mrf.plot.statistics as pltstat


class BackwardEvaluator(evalbase.BaseEvaluator):
    """A backward evaluator, evaluating the goodness of the MR parameter estimation."""

    def __init__(self, reference, prediction, ranges,
                 metrics=metric.get_backward_metrics(), prefix=''):
        """

        Args:
            reference: The reference MR parameters (normalized).
            prediction: The predicted MR parameters (normalized).
            ranges: The original range of the MR parameters before normalization.
            metrics: A list of pymia.evaluation.metric.INumpyArrayMetric.
            prefix: The identifier for the usage of multiple backward evaluators
        """
        self.mr_param_ranges = ranges
        self.reference = norm.de_normalize_mr_parameters(reference, self.mr_param_ranges, defs.MR_PARAMS)
        self.prediction = norm.de_normalize_mr_parameters(prediction, self.mr_param_ranges, defs.MR_PARAMS)

        self.metrics = metrics
        self.prefix = prefix if len(prefix) == 0 else prefix + '_'
        self.results = self.calculate()

    def calculate(self):
        results = {}
        for metric_ in self.metrics:
            mean = []
            for idx, mr_param in enumerate(defs.MR_PARAMS):
                metric_.reference = self.reference[:, idx]
                metric_.prediction = self.prediction[:, idx]
                val = metric_.calculate()
                results[f'{defs.trim_param(mr_param)}/{self.prefix}{metric_.metric}'] = val
                mean.append(val)
            results[f'MEAN/{self.prefix}{metric_.metric}'] = np.mean(mean)
        return results

    def plot(self, root_dir: str):
        for idx, mr_param in enumerate(defs.MR_PARAMS):
            data_ref = self.reference[:, idx]
            data_pred = self.prediction[:, idx]
            map_ = defs.trim_param(mr_param)
            unit = pltlbl.get_map_description(mr_param, True)

            pltstat.bland_altman_plot(os.path.join(root_dir, f'bland-altman-{self.prefix}{map_}.png'), data_pred, data_ref, unit)

            pltstat.scatter_plot(os.path.join(root_dir, f'scatter-{self.prefix}{map_}.png'), data_ref, data_pred,
                                 f'Reference {unit}', f'Predicted {unit}', with_regression_line=True, with_abline=True)

            pltstat.residual_plot(os.path.join(root_dir, f'residual-{self.prefix}{map_}.png'), data_pred, data_ref,
                                  f'Predicted {unit}', f'Residual {unit}')

            pltparam.prediction_distribution_plot(os.path.join(root_dir, f'prediction-{self.prefix}{map_}.png'),
                                                  data_ref, data_pred, unit)

    def save(self, root_dir: str):
        np.save(os.path.join(root_dir, f'mr_parameters_{self.prefix}ref.npy'), self.reference)
        np.save(os.path.join(root_dir, f'mr_parameters_{self.prefix}pred.npy'), self.prediction)
