import numpy as np
import pymia.evaluation.metric as pymia_metric


def get_backward_metrics():
    return [pymia_metric.MeanAbsoluteError(),
            pymia_metric.MeanSquaredError(), pymia_metric.RootMeanSquaredError(),
            pymia_metric.NormalizedRootMeanSquaredError(), pymia_metric.CoefficientOfDetermination(),
            MeanAbsolutePercentageError(), MeanPercentageError()]


class StdAbsoluteError(pymia_metric.NumpyArrayMetric):

    def __init__(self, metric: str = 'STDAE'):
        super().__init__(metric)

    def calculate(self):
        return np.std(np.abs(self.reference - self.prediction))


class MeanAbsolutePercentageError(pymia_metric.NumpyArrayMetric):

    def __init__(self):
        super().__init__()
        self.metric = 'MAPE'

    def calculate(self):
        reference = np.extract(self.reference != 0, self.reference)
        prediction = np.extract(self.reference != 0, self.prediction)

        return np.mean(np.abs((reference - prediction) / reference)) * 100


class MeanPercentageError(pymia_metric.NumpyArrayMetric):

    def __init__(self):
        super().__init__()
        self.metric = 'MPE'

    def calculate(self):
        reference = np.extract(self.reference != 0, self.reference)
        prediction = np.extract(self.reference != 0, self.prediction)

        return np.mean((reference - prediction) / reference) * 100


# todo: might be added in as NumpyArrayMetric. Didn't have the energy...
def relative_error(prediction, reference, eps=1e-8):
    rel_error = np.abs((prediction - reference + eps) / (reference + eps))
    return rel_error
