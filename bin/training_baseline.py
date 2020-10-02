import argparse
import os
import shutil
import threading

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import yaml

import mrf.data.dataset as ds
import mrf.data.definition as defs
import mrf.evaluation.backward as evalbwd
import mrf.loop.callback as clb
import mrf.loop.context as ctx
import mrf.loop.interaction as inter
import mrf.loop.loops as loop
import mrf.loop.utils as utils
import mrf.model.cohen as cohen
import mrf.model.hoppe as hoppe
import mrf.model.invnet as invnet
import mrf.model.oksuz as oksuz
import mrf.model.song as song
import mrf.utilities.logging as log
import mrf.utilities.timestamp as ts


def main(config_file: str):
    with open(config_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # check if model is supported
    model = params["model"]
    supported_models = {
        'invbwd': 'Invertible NN (only backward)',
        'cohen': 'Cohen et al.',
        'oksuz': 'Oksuz et al.',
        'song': 'Song et al.',
        'hoppe': 'Hoppe et al.'
    }

    if model not in supported_models:
        raise ValueError(f'Model "{model}" not supported. Use ' + ', '.join([f'"{k}" ({v})' for k, v in supported_models.items()]))

    run_dir = os.path.join(params['train_dir'],
                           f'{ts.get_timestamp()}_{params["model"]}{"_" + params["experiment"] if params["experiment"] else ""}')
    checkpoint_path = None
    if 'resume' in params:
        checkpoint_path = params['resume']
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    tb = tensorboard.SummaryWriter(run_dir)
    log.setup_file_logging(os.path.join(run_dir, 'log.txt'))
    shutil.copyfile(config_file, os.path.join(run_dir, 'config.yaml'))

    validation_dir = os.path.join(run_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)  # exist_ok for the 'resume' case

    chk_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(chk_dir, exist_ok=True)

    context = MyContext(params, tb)
    interaction = MyInteractions(params, validation_dir)
    callback = clb.ComposeCallback([
        clb.ConsoleLog(), clb.TensorBoardLog(), clb.SaveBest(chk_dir, 'val_loss', higher_better=False),
        clb.SaveBest(chk_dir, 'MEAN/R2')
    ])

    trainer = loop.TrainLoop(context, interaction, callback, epochs=params['epochs'], seed=params['seed'], only_validate=False)
    trainer.train(checkpoint_path)

    tb.close()


class MyContext(ctx.Context):

    def __init__(self, params, tb: tensorboard.SummaryWriter) -> None:
        super().__init__()
        self.params = params
        self._device = 'cuda'
        self.tb = tb

        # model configuration
        self.model_name = self.params['model']
        self.no_mr_params = self.params['no_mr_params']  # number of MR parameters
        self.no_frames = self.params['no_frames']  # number of MRF frames (network input will be twice no_frames due to real and imag parts)

    def _init_model(self):
        ch_in = self.no_frames * 2  # concatenation of real and imaginary parts
        ch_out = self.no_mr_params
        if self.model_name == cohen.MODEL_COHEN:
            return cohen.CohenModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
        elif self.model_name == hoppe.MODEL_HOPPE:
            return hoppe.HoppeCNNModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
        elif self.model_name == oksuz.MODEL_OKSUZ:
            return oksuz.OksuzModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
        elif self.model_name == song.MODEL_SONG:
            return song.SongModel(ch_in=ch_in, ch_out=ch_out).to(self.device)
        elif self.model_name == 'invbwd':
            model_params = self.params['model_params'] if 'model_params' in self.params else {}
            return invnet.get_invnet(ndim=ch_in, **model_params).to(self.device)
        else:
            raise ValueError(f'Unknown model "{self.model_name}".')

    def _init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

    def _init_train_loader(self):
        dataset = ds.NumpyMRFDataset(self.params['train_database_dir'])
        loader = data.DataLoader(dataset, batch_size=self.params['batch_size_training'], shuffle=True, num_workers=0)
        return loader

    def _init_valid_loader(self):
        dataset = ds.NumpyMRFDataset(self.params['valid_database_dir'])
        loader = data.DataLoader(dataset, batch_size=self.params['batch_size_testing'], shuffle=False, num_workers=0)
        return loader

    @property
    def device(self):
        return self._device


class MyInteractions(inter.Interaction):

    def __init__(self, params, validation_dir) -> None:
        super().__init__()
        self.validation_dir = validation_dir
        self.no_mr_params = params['no_mr_params']
        self.no_frames = params['no_frames']

        self.ndim_y = self.no_frames * 2  # concatenation of real and imaginary parts
        self.y_noise_scale = params['y_noise']
        self.y_noise_scale_validation = params['y_noise_validation']

    def training_step(self, context: MyContext, batch, loop_info):
        context.optimizer.zero_grad()
        x, y = batch[defs.KEY_MR_PARAMS].float().to(context.device), batch[defs.KEY_FINGERPRINTS].float().to(context.device)

        if self.y_noise_scale:
            y += self.y_noise_scale * torch.randn(y.size(0), self.ndim_y, device=y.device)

        x_hat = context.model(y)[:, :self.no_mr_params]
        loss = F.mse_loss(x_hat, x)

        loss.backward()
        context.optimizer.step()

        return {'loss': loss.item(), 'recon_bwd': loss.item()}

    def validation_step(self, context: MyContext, batch, loop_info):
        x, y = batch[defs.KEY_MR_PARAMS].float().to(context.device), batch[defs.KEY_FINGERPRINTS].float().to(context.device)
        x_ref = x.cpu().numpy()

        x_hat = context.model(y)[:, :self.no_mr_params]
        loss = F.mse_loss(x_hat, x)

        # now perturb with noise for additional validation
        y = y + self.y_noise_scale_validation * torch.randn(y.size(0), self.ndim_y, device=y.device)
        x_hat_noisy = context.model(y)[:, :self.no_mr_params]

        results = {
            'val_loss': loss.item(),
            'reference_mr_params': x_ref,
            'prediction_mr_params': x_hat.cpu().numpy(),
            'prediction_mr_params_noisy': x_hat_noisy.cpu().numpy(),
        }

        return results

    def validation_summary(self, context: MyContext, results, loop_info):
        accumulation = {}
        for batch_results in results:
            for k, v, in batch_results.items():
                accumulation.setdefault(k, []).append(v)

        epoch_dir = utils.prepare_epoch_result_directory(self.validation_dir, loop_info['epoch'])

        ref_mr_params = np.vstack(accumulation.pop('reference_mr_params'))
        pred_mr_params = np.vstack(accumulation.pop('prediction_mr_params'))
        evaluator_bwd = evalbwd.BackwardEvaluator(ref_mr_params,
                                                  pred_mr_params,
                                                  context.valid_loader.dataset.mr_param_ranges)

        evaluator_bwd_noise = evalbwd.BackwardEvaluator(ref_mr_params,
                                                        np.vstack(accumulation.pop('prediction_mr_params_noisy')),
                                                        context.valid_loader.dataset.mr_param_ranges,
                                                        prefix='NOISE')

        evaluators = [evaluator_bwd, evaluator_bwd_noise]

        # common plot (same thread) since plot issues observed, otherwise
        def plot():
            for e in evaluators:
                e.plot(epoch_dir)
        threading.Thread(target=plot).start()

        summary = {}
        for e in evaluators:
            summary.update(e.results)
        for k, v in accumulation.items():
            summary[k] = np.mean(v)

        return summary

    def get_best(self, context: MyContext, summary, best, loop_info):
        if 'val_loss' not in best or best['val_loss'] > summary['val_loss']:
            best['val_loss'] = summary['val_loss']
        if 'MEAN/R2' not in best or best['MEAN/R2'] < summary['MEAN/R2']:
            best['MEAN/R2'] = summary['MEAN/R2']
        return best


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks')

    parser.add_argument(
        '--config_file',
        type=str,
        default='./config/config_baseline.yaml',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
