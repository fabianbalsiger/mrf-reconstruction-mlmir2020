import csv
import logging
import os
import time

import torch.utils.tensorboard as tensorboard

import mrf.loop.context as ctx
import mrf.loop.utils as utils


class Callback:

    def on_train_start(self, context: ctx.Context, loop_info):
        pass

    def on_train_end(self, context: ctx.Context, loop_info):
        pass

    def on_epoch_start(self, context: ctx.Context, loop_info):
        pass

    def on_epoch_end(self, context: ctx.Context, summary, loop_info):
        pass

    def on_batch_start(self, context: ctx.Context, batch, loop_info):
        pass

    def on_batch_end(self, context: ctx.Context, result, loop_info):
        pass

    def on_validation_start(self, context: ctx.Context, loop_info):
        pass

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        pass

    def on_validation_batch_start(self, context: ctx.Context, batch, loop_info):
        pass

    def on_validation_batch_end(self, context: ctx.Context, result, loop_info):
        pass

    def on_test_start(self, context: ctx.TestContext, loop_info):
        pass

    def on_test_end(self, context: ctx.TestContext, summary, loop_info):
        pass

    def on_test_batch_start(self, context: ctx.TestContext, batch, loop_info):
        pass

    def on_test_batch_end(self, context: ctx.TestContext, result, loop_info):
        pass


def make_reduce_compose(obj, hook_cls, hooks: list):
    """only keeps the overridden methods not the empty ones"""
    def _get_loop_fn(fns):
        def loop(*args, **kwargs):
            for fn in fns:
                fn(*args, **kwargs)
        return loop

    method_list = [func for func in dir(hook_cls)
                   if callable(getattr(hook_cls, func)) and not func.startswith("__")]
    for method in method_list:
        hook_fns = []
        for hook in hooks:
            base_fn = getattr(hook_cls, method)
            hook_fn = getattr(hook, method)
            if hook_fn.__func__ != base_fn:
                hook_fns.append(hook_fn)
        setattr(obj, method, _get_loop_fn(hook_fns))


class ComposeCallback(Callback):

    def __init__(self, callbacks: list) -> None:
        super().__init__()
        make_reduce_compose(self, Callback, callbacks)


class ConsoleLog(Callback):

    def __init__(self, log_every_nth=10, log_every_test=True) -> None:
        super().__init__()
        self.log_every_nth = log_every_nth
        self.train_batch_start_time = None
        self.valid_start_time = None
        self.test_start_time = None
        self.test_batch_start_time = None
        self.log_every_test = log_every_test

    def on_train_start(self, context: ctx.Context, loop_info):
        logging.info('model: \n{}'.format(str(context.model)))
        params = sum(p.numel() for p in context.model.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(params))
        logging.info('startup finished')

        logging.info('loop')
        self.train_batch_start_time = time.time()

    def on_batch_end(self, context: ctx.Context, result, loop_info):
        epochs, epoch = loop_info.get('epochs'),  loop_info.get('epoch')
        batches, batch_idx = loop_info.get('batches'), loop_info.get('batch_idx')

        if ((batch_idx + 1) % self.log_every_nth == 0) or (batch_idx == batches - 1):
            duration = time.time() - self.train_batch_start_time
            result_string = ' | '.join(f'{v}: {k:.2e}' for v, k in result.items())
            logging.info(f'[{epoch + 1}/{epochs}, {batch_idx + 1}/{batches}, {duration:.2f}s] {result_string}')
            # start timing here in order to take into account the data loading
            self.train_batch_start_time = time.time()

    def on_validation_start(self, context: ctx.Context, loop_info):
        logging.info('validating')
        self.valid_start_time = time.time()

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        epochs, epoch = loop_info.get('epochs'),  loop_info.get('epoch')

        duration = time.time() - self.valid_start_time
        result_string = ' | '.join(f'{v}: {k:.2e}' for v, k in summary.items())
        logging.info(f'[{epoch + 1}/{epochs}, {duration:.2f}s] {result_string}')

    def on_test_start(self, context: ctx.TestContext, loop_info):
        logging.info('testing')
        self.test_start_time = time.time()

    def on_test_batch_start(self, context: ctx.TestContext, batch, loop_info):
        self.test_batch_start_time = time.time()

    def on_test_batch_end(self, context: ctx.TestContext, result, loop_info):
        batch_end_time = time.time()
        duration = batch_end_time - self.test_batch_start_time
        if self.log_every_test:
            logging.info(f'[{loop_info["batch_idx"] + 1}/{loop_info["batches"]}, {duration:.2f}s]')
        self.test_batch_start_time = batch_end_time

    def on_test_end(self, context: ctx.TestContext, summary, loop_info):
        duration = time.time() - self.test_start_time
        result_string = ' | '.join(f'{v}: {k:.2e}' for v, k in summary.items())
        logging.info(f'[{duration:.2f}s] {result_string}')


class TensorBoardLog(Callback):

    def __init__(self, log_every_nth=1) -> None:
        super().__init__()
        self.log_every_nth = log_every_nth
        self.start_time_epoch = 0
        self.start_time_valid = 0

    def on_epoch_start(self, context: ctx.Context, loop_info):
        self.start_time_epoch = time.time()

    def on_epoch_end(self, context: ctx.Context, summary, loop_info):
        duration = time.time() - self.start_time_epoch

        tb = self._get_logger_or_raise_error(context)
        epoch = loop_info.get('epoch')

        tb.add_scalar('train/time', duration, epoch)

    def on_validation_start(self, context: ctx.Context, loop_info):
        self.start_time_valid = time.time()

    def on_batch_end(self, context: ctx.Context, result, loop_info):
        tb = self._get_logger_or_raise_error(context)

        epochs, epoch = loop_info.get('epochs'),  loop_info.get('epoch')
        batches, batch_idx = loop_info.get('batches'), loop_info.get('batch_idx')

        if ((batch_idx + 1) % self.log_every_nth == 0) or (batch_idx == batches - 1):
            step = epoch * batches + batch_idx
            for key, result in result.items():
                tb.add_scalar('train/{}'.format(key), result, step)

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        duration = time.time() - self.start_time_epoch

        tb = self._get_logger_or_raise_error(context)
        epoch = loop_info.get('epoch')

        for key, result, in summary.items():
            tb.add_scalar('valid/{}'.format(key), result, epoch)

        tb.add_scalar('valid/time', duration, epoch)

    @staticmethod
    def _get_logger_or_raise_error(context: ctx.Context) -> tensorboard.SummaryWriter:
        if not hasattr(context, 'tb'):
            raise ValueError('missing TensorBoard logger in context')
        return context.tb


class SaveBest(Callback):

    def __init__(self, checkpoint_dir, metric, higher_better=True) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.metric = metric
        self.higher_better = higher_better
        self.saved_best = None
        self.saved_best_path = None

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        if self.metric not in best:
            raise ValueError(f'{self.metric} missing as entry of "best"')

        if (self.saved_best is None) or \
                (self.higher_better and best[self.metric] > self.saved_best) or \
                (not self.higher_better and best[self.metric] < self.saved_best):

            # first delete the existing
            if self.saved_best_path is not None and os.path.isfile(self.saved_best_path):
                os.remove(self.saved_best_path)

            epoch = loop_info.get('epoch')
            checkpoint_path = utils.get_checkpoint_path(self.checkpoint_dir, epoch, self._clean_metric_for_path(self.metric))
            context.save_checkpoint(checkpoint_path, epoch, best)
            self.saved_best = best[self.metric]
            self.saved_best_path = checkpoint_path

    @staticmethod
    def _clean_metric_for_path(metric_name):
        return metric_name.replace('/', '-')


class SaveNLast(Callback):

    def __init__(self, checkpoint_dir: str, n_last: int) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.n_last = n_last

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        epoch = loop_info.get('epoch')
        to_remove = epoch - self.n_last
        if to_remove >= 0:
            checkpoint_path = utils.get_checkpoint_path(self.checkpoint_dir, to_remove)
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)

        checkpoint_path = utils.get_checkpoint_path(self.checkpoint_dir, epoch)
        context.save_checkpoint(checkpoint_path, epoch, best)


class SaveTestResults(Callback):

    def __init__(self, out_dir: str):
        super().__init__()
        self.path = out_dir

    def on_test_end(self, context: ctx.TestContext, summary, loop_info):
        maps = list(set([k.split('/')[0] for k in summary.keys()]))
        maps.sort()
        metrics = list(set([k.split('/')[1] for k in summary.keys()]))
        metrics.sort()

        with open(os.path.join(self.path, 'results_readable.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['MAP', ] + metrics)
            for map_ in maps:
                row = [map_]
                for metric in metrics:
                    key = f'{map_}/{metric}'
                    if key in summary:
                        row.append(summary[key])
                    else:
                        row.append('NA')
                writer.writerow(row)

        with open(os.path.join(self.path, 'results.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['METRIC', 'VALUE'])
            for k, v in summary.items():
                writer.writerow([k, v])
