import argparse
import csv
import logging
import os
import shutil
import threading

import numpy as np
import torch
import torch.utils.data as data
import yaml

import mrf.data.dataset as ds
import mrf.data.definition as defs
import mrf.data.normalization as norm
import mrf.loop.callback as clb
import mrf.loop.context as ctx
import mrf.loop.interaction as inter
import mrf.loop.loops as loop
import mrf.model.cohen as cohen
import mrf.model.hoppe as hoppe
import mrf.model.invnet as invnet
import mrf.model.oksuz as oksuz
import mrf.model.song as song
import mrf.evaluation.backward as evalbwd
import mrf.evaluation.forward as evalfwd
import mrf.evaluation.metric as metric
import mrf.utilities.logging as log
import mrf.utilities.timestamp as ts


def main(config_file: str):
    with open(config_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    checkpoint_path = params['model_path']
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    # get training configuration for model
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        params_training = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in params_training.items():
        if k not in params:
            params[k] = v

    experiment = params['experiment']
    if os.path.isdir(experiment):
        test_dir = experiment
    else:
        test_dir = os.path.join(run_dir, 'test', f'{ts.get_timestamp()}{"_" + experiment if experiment else ""}')
        os.makedirs(test_dir)

    def prepare_subdir(sub_directory):
        os.makedirs(sub_directory)
        log.setup_file_logging(os.path.join(sub_directory, 'log.txt'))
        shutil.copyfile(config_file, os.path.join(sub_directory, 'config.yaml'))

    if params['do_basic_test']:
        sub_dir = os.path.join(test_dir, 'bwdfwd')
        prepare_subdir(sub_dir)

        y_noise_levels = params["y_noise_levels"]
        logging.info('\n\n----- Test Model ------')
        logging.info(f'\t- Noise levels: {y_noise_levels}')

        collector = TestInteraction.CollectorCallback(y_noise_levels)
        callback = clb.ComposeCallback([
            clb.ConsoleLog(log_every_test=False),
            collector
        ])
        context = MyContext(params)
        interaction = TestInteraction(params, sub_dir)
        for i, y_noise in enumerate(y_noise_levels):
            logging.info(f'\t- [{i + 1}/{len(y_noise_levels)}] level: {y_noise}')
            interaction.set_y_noise_scale(y_noise)
            tester = loop.Tester(context, interaction, callback)
            tester.test(checkpoint_path)
        collector.save_history(sub_dir, params['out_file_postfix'])

    if params['do_snr']:
        snr_dir = os.path.join(test_dir, 'snr')
        prepare_subdir(snr_dir)

        snr_y_noise_levels = params["snr_y_noise_levels"]
        logging.info('\n\n----- Test SNR ------')
        logging.info(f'\t- Noise levels: {snr_y_noise_levels}')
        logging.info(f'\t- SNRs: {params["snrs"]}')

        collector = SNRInteractions.CollectorCallback(params['snrs'])
        callback = clb.ComposeCallback([
            clb.ConsoleLog(log_every_test=False),
            collector
        ])
        context = MyContext(params)
        interaction = SNRInteractions(params)

        for i, y_noise in enumerate(snr_y_noise_levels):
            logging.info(f'\t- [{i+1}/{len(snr_y_noise_levels)}] level: {y_noise}')
            interaction.set_y_noise_scale(y_noise)
            tester = loop.Tester(context, interaction, callback)
            tester.test(checkpoint_path)
        collector.save_history(snr_dir, params['out_file_postfix'])


class MyContext(ctx.TestContext):

    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self._device = 'cuda'
        self.wrapper = None

    def _init_model(self):
        ch_in = self.params['no_frames'] * 2
        ch_out = self.params['no_mr_params']

        if self.params['model'] == 'invfwdbwd':
            model_params = self.params['model_params'] if 'model_params' in self.params else {}
            model = invnet.get_invnet(ch_in, **model_params).to(self.device)
            self.wrapper = INNWrapper(model, self.params['no_mr_params'], ch_in)
            return model
        elif self.params['model'] == cohen.MODEL_COHEN:
            model = cohen.CohenModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
            self.wrapper = BaselineWrapper(model, self.params['no_mr_params'])
        elif self.params['model'] == hoppe.MODEL_HOPPE:
            model = hoppe.HoppeCNNModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
            self.wrapper = BaselineWrapper(model, self.params['no_mr_params'])
        elif self.params['model'] == oksuz.MODEL_OKSUZ:
            model = oksuz.OksuzModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
            self.wrapper = BaselineWrapper(model, self.params['no_mr_params'])
        elif self.params['model'] == song.MODEL_SONG:
            model = song.SongModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
            self.wrapper = BaselineWrapper(model, self.params['no_mr_params'])
        elif self.params['model'] == 'invbwd':
            # ndim initialization is misleading due to "misuse" of InvNet for fair comparison
            model_params = self.params['model_params'] if 'model_params' in self.params else {}
            model = invnet.get_invnet(ch_in, **model_params).to(self.device)
            self.wrapper = BaselineWrapper(model, self.params['no_mr_params'])
        else:
            raise ValueError(f'Unknown model "{self.params["model"]}".')
        return model

    def _init_test_loader(self):
        dataset = ds.NumpyMRFDataset(self.params['test_database_dir'])
        loader = data.DataLoader(dataset, batch_size=self.params['batch_size_testing'], shuffle=False, num_workers=0)
        return loader

    @property
    def device(self):
        return self._device


class SNRInteractions(inter.TestInteraction):

    class CollectorCallback(clb.Callback):

        def __init__(self, snr_levels) -> None:
            super().__init__()
            self.snr_levels = snr_levels
            self.history = {}

        def on_test_end(self, context: ctx.TestContext, summary, loop_info):
            for k, v in summary.items():
                self.history.setdefault(k, []).append(v)

        def save_history(self, out_dir, postfix=''):
            if len(postfix) > 0:
                postfix = '_' + postfix
            out_file = os.path.join(out_dir, f'snr_results{postfix}.csv')
            with open(out_file, 'w') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['MAP', 'METRIC', 'SNR', 'MEAN', 'STD'])

                for k, v in self.history.items():
                    map_, metric, type_ = k.split('/')
                    # todo: VERY DIRTY hack here. but who cares...
                    if type_ == 'STD':
                        continue
                    else:
                        means = v
                        stds = self.history[f'{map_}/{metric}/STD']

                    for idx, snr in enumerate(self.snr_levels):
                        writer.writerow([map_, metric, snr, means[idx], stds[idx]])

    def __init__(self, params) -> None:
        super().__init__()
        self.noise_samples = params['noise_samples']
        self.y_noise_scale = None

    def set_y_noise_scale(self, scale):
        self.y_noise_scale = scale

    def test_step(self, context: MyContext, batch, loop_info):
        assert self.y_noise_scale is not None

        x, y = batch[defs.KEY_MR_PARAMS].float().to(context.device), batch[defs.KEY_FINGERPRINTS].float().to(
            context.device)

        x_clean, y_clean = x.clone(), y.clone()
        x_ref = x_clean.cpu().numpy()

        noisy_predictions = []
        results = {}
        for i in range(self.noise_samples):
            y_noisy = y_clean + self.y_noise_scale * torch.randn_like(y_clean)
            x_pred_noisy = context.wrapper.predict(y_noisy).cpu().numpy()
            noisy_predictions.append(x_pred_noisy)

        results['x_hat_samples'] = np.stack(noisy_predictions, axis=-1)
        results['x_ref'] = x_ref
        return results

    def test_summary(self, context: MyContext, results, loop_info):
        accumulation = {}
        for batch_results in results:
            for k, v, in batch_results.items():
                accumulation.setdefault(k, []).append(v)

        noise_samples = np.vstack(accumulation.pop('x_hat_samples'))
        x_ref = np.vstack(accumulation.pop('x_ref'))

        noise_samples = norm.de_normalize_mr_parameters(noise_samples, context.test_loader.dataset.mr_param_ranges)
        x_ref = norm.de_normalize_mr_parameters(x_ref, context.test_loader.dataset.mr_param_ranges)

        summary = {}

        for idx, mr_param in enumerate(defs.MR_PARAMS):
            ref = x_ref[:, idx:idx + 1]  # slice because we want shape (:, 1) not (:,)
            rel_error = metric.relative_error(noise_samples[:, idx], ref)
            # todo: the relative error is still for each fingerprint and all the samples (no_fp, no_samples)
            # what to calculate mean and std from?
            summary[f'{defs.trim_param(mr_param)}/REL_ERR/MEAN'] = rel_error.mean()
            summary[f'{defs.trim_param(mr_param)}/REL_ERR/STD'] = rel_error.std()

        return summary


class TestInteraction(inter.TestInteraction):

    class CollectorCallback(clb.Callback):

        def __init__(self, y_noise_levels) -> None:
            super().__init__()
            self.y_noise_levels = y_noise_levels
            self.history = {}

        def on_test_end(self, context: MyContext, summary, loop_info):
            for k, v in summary.items():
                self.history.setdefault(k, []).append(v)

        def save_history(self, out_dir, postfix=''):
            if len(postfix) > 0:
                postfix = '_' + postfix
            out_file = os.path.join(out_dir, f'test_results{postfix}.csv')
            with open(out_file, 'w') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['MAP', 'METRIC', 'NOISE', 'MEAN', 'STD'])

                for k, v in self.history.items():
                    parts = k.split('/')
                    map_, metric_ = parts[0:2]
                    # todo: and once more: VERY DIRTY hack here. but who cares...
                    if len(parts) > 2 and parts[2] == 'STD':
                        continue
                    else:
                        means = v
                        stds = self.history.get(f'{map_}/{metric_}/STD', None)

                    for idx, y_noise in enumerate(self.y_noise_levels):
                        writer.writerow([map_, metric_, y_noise, means[idx], stds[idx] if stds else 'N/A'])

    def __init__(self, params, test_dir) -> None:
        super().__init__()
        self.root_dir = test_dir
        self.save_fingerprints_if_fwd_known = params['save_fingerprints_if_fwd_known']
        self.test_dir = None
        self.y_noise_scale = None

    def set_y_noise_scale(self, scale):
        self.y_noise_scale = scale
        self.test_dir = os.path.join(self.root_dir, f'{self.y_noise_scale:.1e}')
        os.makedirs(self.test_dir)

    def test_step(self, context: MyContext, batch, loop_info):
        assert self.y_noise_scale is not None
        x, y = batch[defs.KEY_MR_PARAMS].float().to(context.device), batch[defs.KEY_FINGERPRINTS].float().to(context.device)

        x_clean, y_clean = x.clone(), y.clone()
        x_ref = x_clean.cpu().numpy()

        y = y_clean + self.y_noise_scale * torch.randn_like(y_clean)

        x_pred = context.wrapper.predict(y).cpu().numpy()
        results = {
            'x_pred': x_pred,
            'x_ref': x_ref
        }

        if context.wrapper.can_predict_y():
            y_pred = context.wrapper.predict_y(x).cpu().numpy()
            results['y_pred'] = y_pred
            results['y_ref'] = y_clean.cpu().numpy()

        return results

    def test_summary(self, context: MyContext, results, loop_info):
        accumulation = {}
        for batch_results in results:
            for k, v, in batch_results.items():
                accumulation.setdefault(k, []).append(v)

        x_pred = np.vstack(accumulation.pop('x_pred'))
        x_ref = np.vstack(accumulation.pop('x_ref'))
        backward_evaluator = evalbwd.BackwardEvaluator(x_ref, x_pred,
                                                       context.test_loader.dataset.mr_param_ranges,
                                                       metrics=[metric.pymia_metric.MeanAbsoluteError('MAE/MEAN'),
                                                                metric.StdAbsoluteError('MAE/STD'),
                                                                metric.pymia_metric.CoefficientOfDetermination()])
        evaluators = [backward_evaluator]

        if 'y_pred' in accumulation:
            y_pred = np.vstack(accumulation.pop('y_pred'))
            y_ref = np.vstack(accumulation.pop('y_ref'))
            corr_evaluator = evalfwd.ForwardCorrelationEvaluator(y_ref, y_pred, x_ref, x_pred,
                                                                 context.test_loader.dataset.mr_param_ranges,
                                                                 self.save_fingerprints_if_fwd_known)
            evaluators.append(corr_evaluator)

        plot_and_save_in_background(evaluators, self.test_dir, context)

        summary = {}
        for e in evaluators:
            summary.update(e.results)
        return summary


def plot_and_save_in_background(evaluators, test_dir, context):
    if context.params['plot_results']:
        def plot():
            for e in evaluators:
                e.plot(test_dir)
        threading.Thread(target=plot).start()

    if context.params['save_data']:
        def save():
            for e in evaluators:
                e.save(test_dir)
        threading.Thread(target=save).start()


class ModelWrapper:

    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict(self, y):
        raise NotImplementedError()

    def can_predict_y(self):
        return False


class INNWrapper(ModelWrapper):

    def __init__(self, model, no_params, ndim_y):
        super().__init__(model)
        self.no_params = no_params
        self.ndim_y = ndim_y

    def predict(self, y):
        return self.model(y, rev=True)[:, :self.no_params]

    def can_predict_y(self):
        return True

    def predict_y(self, x):
        batch_size = x.size(0)
        x = torch.cat((x, torch.zeros(batch_size, self.ndim_y - self.no_params, device=x.device)), dim=1)
        y_pred = self.model(x)[:, -self.ndim_y:]
        return y_pred


class BaselineWrapper(ModelWrapper):

    def __init__(self, model, no_params):
        super().__init__(model)
        self.no_params = no_params

    def predict(self, y):
        return self.model(y)[:, :self.no_params]


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks')

    parser.add_argument(
        '--config_file',
        type=str,
        default='./config/config_test.yaml',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
