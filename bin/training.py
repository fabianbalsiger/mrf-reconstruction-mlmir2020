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
import mrf.evaluation.forward as evalfwd
import mrf.loop.callback as clb
import mrf.loop.context as ctx
import mrf.loop.interaction as inter
import mrf.loop.loops as loop
import mrf.loop.utils as utils
import mrf.model.invnet as invnet
import mrf.utilities.logging as log
import mrf.utilities.timestamp as ts


def main(config_file: str):
    with open(config_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    checkpoint_path = None
    if 'resume' in params:
        checkpoint_path = params['resume']
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        shutil.copyfile(config_file, os.path.join(run_dir, 'resume_config.yaml'))

        # get training configuration and add additional epochs to be calculated
        with open(os.path.join(run_dir, 'config.yaml')) as f:
            orig_params = yaml.load(f, Loader=yaml.FullLoader)
            orig_params['epochs'] += params['epochs']  # add the additional epochs to be calculated
            params = orig_params
    else:
        run_dir = os.path.join(params['train_dir'], f'{ts.get_timestamp()}_{params["model"]}'f'{"_" + params["experiment"] if params["experiment"] else ""}')
        os.makedirs(run_dir)
        shutil.copyfile(config_file, os.path.join(run_dir, 'config.yaml'))

    tb = tensorboard.SummaryWriter(run_dir)
    log.setup_file_logging(os.path.join(run_dir, 'log.txt'))

    validation_dir = os.path.join(run_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)  # exist_ok for the 'resume' case

    chk_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(chk_dir, exist_ok=True)

    context = MyContext(params, tb)
    interaction = MyInteractions(params, validation_dir)
    callback = clb.ComposeCallback([
        clb.ConsoleLog(), clb.TensorBoardLog(10), clb.SaveBest(chk_dir, 'val_recon_bwd_nw', higher_better=False),
        clb.SaveBest(chk_dir, 'MEAN/R2'),
        clb.SaveBest(chk_dir, 'MEAN/NOISE_R2')
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

    def _init_model(self):
        ndim = self.params['no_frames'] * 2  # concatenation of real and imaginary parts
        model_params = self.params['model_params'] if 'model_params' in self.params else {}
        model = invnet.get_invnet(ndim, **model_params)
        return model.to(self.device)

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

        self.ndim_x = params['no_mr_params']
        self.ndim_y = params['no_frames'] * 2  # concatenation of real and imaginary parts

        self.y_noise_scale = params['y_noise']
        self.y_noise_scale_validation = params['y_noise_validation']

        self.lambd_recon_fwd = params['weight_recon_fwd']
        self.lambd_recon_bwd = params['weight_recon_bwd']

    def training_step(self, context: MyContext, batch, loop_info):
        x, y = batch[defs.KEY_MR_PARAMS].float().to(context.device), batch[defs.KEY_FINGERPRINTS].float().to(context.device)
        batch_size = x.size(0)

        x_clean, y_clean = x.clone(), y.clone()

        if self.y_noise_scale:
            y += self.y_noise_scale * torch.randn(batch_size, self.ndim_y, dtype=torch.float, device=y.device)

        pad_x = torch.zeros(batch_size, self.ndim_y - self.ndim_x, device=x.device)
        x = torch.cat((x, pad_x), dim=1)

        context.optimizer.zero_grad()

        # Forward step
        y_hat = context.model(x)
        loss_recon_fwd = F.mse_loss(y_hat, y)
        loss_recon_fwd_w = self.lambd_recon_fwd * loss_recon_fwd
        loss_recon_fwd_w.backward()

        # Backward step
        if self.y_noise_scale:
            y = y_clean + self.y_noise_scale * torch.randn(batch_size, self.ndim_y, device=x.device)

        x_hat = context.model(y, rev=True)
        loss_recon_bwd = F.mse_loss(x_hat, x)
        loss_recon_bwd_w = self.lambd_recon_bwd * loss_recon_bwd
        loss_recon_bwd_w.backward()

        for p in context.model.parameters():
            p.grad.data.clamp_(-15.00, 15.00)

        context.optimizer.step()

        results = {'loss': (loss_recon_fwd_w + loss_recon_bwd_w).item(),
                   'recon_fwd': loss_recon_fwd_w.item(),
                   'recon_bwd': loss_recon_bwd_w.item(),
                   'recon_fwd_nw': loss_recon_fwd.item(),
                   'recon_bwd_nw': loss_recon_bwd.item(),
                   }

        return results

    def validation_step(self, context: MyContext, batch, loop_info):
        x, y = batch[defs.KEY_MR_PARAMS].float().to(context.device), batch[defs.KEY_FINGERPRINTS].float().to(context.device)
        batch_size = x.size(0)

        x_clean, y_clean = x.clone(), y.clone()

        if self.y_noise_scale:
            y += self.y_noise_scale * torch.randn(batch_size, self.ndim_y, dtype=torch.float, device=y.device)

        pad_x = torch.zeros(batch_size, self.ndim_y - self.ndim_x, device=x.device)
        x = torch.cat((x, pad_x), dim=1)

        # Forward step
        y_hat = context.model(x)
        loss_recon_fwd = F.mse_loss(y_hat, y)

        # Backward step
        if self.y_noise_scale:
            y = y_clean + self.y_noise_scale * torch.randn(batch_size, self.ndim_y, device=x.device)
        x_hat = context.model(y, rev=True)
        loss_recon_bwd = F.mse_loss(x_hat, x)

        # actual prediction
        x = torch.cat((x_clean, torch.zeros(batch_size, self.ndim_y - self.ndim_x, device=x.device)), dim=1)
        y_pred = context.model(x)
        y_recon = F.mse_loss(y_pred, y_clean)

        x_pred = context.model(y_clean, rev=True)[:, :self.ndim_x]
        x_recon = F.mse_loss(x_pred, x_clean)

        # now perturb with noise for validation too
        y = y_clean + self.y_noise_scale_validation * torch.randn(batch_size, self.ndim_y, device=x.device)
        x_hat_noisy = context.model(y, rev=True)[:, :self.ndim_x]

        results = {
            'val_recon_fwd_nw': loss_recon_fwd.item(),
            'val_recon_bwd_nw': loss_recon_bwd.item(),

            'val_recon_bwd_pred': x_recon.item(),
            'val_recon_fwd_pred': y_recon.item(),

            'reference_mr_params': x_clean.cpu().numpy(),
            'prediction_mr_params': x_pred.cpu().numpy(),
            'prediction_mr_params_noisy': x_hat_noisy.cpu().numpy(),
            'reference_fingerprints': y_clean.cpu().numpy(),
            'prediction_fingerprints': y_pred.cpu().numpy(),
        }
        return results

    def validation_summary(self, context: MyContext, results, loop_info):
        accumulation = {}
        for batch_results in results:
            for k, v, in batch_results.items():
                accumulation.setdefault(k, []).append(v)

        ref_mr_params = np.vstack(accumulation.pop('reference_mr_params'))
        epoch_dir = utils.prepare_epoch_result_directory(self.validation_dir, loop_info['epoch'])

        evaluator_bwd = evalbwd.BackwardEvaluator(ref_mr_params,
                                                  np.vstack(accumulation.pop('prediction_mr_params')),
                                                  context.valid_loader.dataset.mr_param_ranges)

        evaluator_bwd_noise = evalbwd.BackwardEvaluator(ref_mr_params,
                                                        np.vstack(accumulation.pop('prediction_mr_params_noisy')),
                                                        context.valid_loader.dataset.mr_param_ranges,
                                                        prefix='NOISE')

        evaluator_fwd = evalfwd.ForwardEvaluator(np.vstack(accumulation.pop('reference_fingerprints')),
                                                 np.vstack(accumulation.pop('prediction_fingerprints')),
                                                 ref_mr_params,
                                                 context.valid_loader.dataset.mr_param_ranges)

        evaluators = [evaluator_bwd, evaluator_fwd, evaluator_bwd_noise]

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
        if 'val_recon_bwd_nw' not in best or best['val_recon_bwd_nw'] > summary['val_recon_bwd_nw']:
            best['val_recon_bwd_nw'] = summary['val_recon_bwd_nw']
        if 'MEAN/R2' not in best or best['MEAN/R2'] < summary['MEAN/R2']:
            best['MEAN/R2'] = summary['MEAN/R2']
        if 'MEAN/NOISE_R2' not in best or best['MEAN/NOISE_R2'] < summary['MEAN/NOISE_R2']:
            best['MEAN/NOISE_R2'] = summary['MEAN/NOISE_R2']

        return best


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks')

    parser.add_argument(
        '--config_file',
        type=str,
        default='./config/config.yaml',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
