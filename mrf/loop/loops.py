import mrf.loop.callback as clb
import mrf.loop.context as ctx
import mrf.loop.interaction as inter
import mrf.utilities.seeding as seeding


class TrainLoop:

    def __init__(self, context: ctx.Context, interaction: inter.Interaction, callback: clb.Callback, epochs, seed=None,
                 validate_every=1, only_validate=False) -> None:
        super().__init__()
        self.context = context
        self.interaction = interaction
        self.callback = callback
        self.epochs = epochs
        self.seed = seed
        self.cudnn_seed = False
        self.validate_every = validate_every
        self.only_validate = only_validate

        self.best = {}
        self.epoch = 0

        if self.seed is not None:
            seeding.do_seed(self.seed, with_cudnn=self.cudnn_seed)
        # by purpose after seed since initialization depends on random values
        self.context.setup()

    def train(self, checkpoint: str = None):
        if checkpoint is not None:
            self.epoch, self.best = self.context.load_checkpoint(checkpoint)
            self.epoch += 1  # continue at next checkpoint

        loop_info = {'epochs': self.epochs, 'batches': len(self.context.train_loader)}

        self.callback.on_train_start(self.context, self.epochs)
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            loop_info['epoch'] = epoch

            if not self.only_validate:
                if self.seed is not None and epoch > 0:
                    # seed every epoch such that same loop can also be guaranteed at loop continuation
                    seeding.do_seed(self.seed + epoch, with_cudnn=self.cudnn_seed)

                self.callback.on_epoch_start(self.context, loop_info)
                self.interaction.epoch_start(self.context, loop_info)
                results = []
                for batch_idx, batch in enumerate(self.context.train_loader):
                    loop_info['batch_idx'] = batch_idx
                    self.callback.on_batch_start(self.context, batch, loop_info)
                    result = self.interaction.training_step(self.context, batch, loop_info)
                    self.callback.on_batch_end(self.context, result, loop_info)
                    results.append(result)

                del loop_info['batch_idx']
                summary = self.interaction.train_summary(self.context, results, loop_info)
                self.callback.on_epoch_end(self.context, summary, loop_info)
            if (epoch + 1) % self.validate_every == 0:
                self.validate()

        del loop_info['epoch']
        self.callback.on_train_end(self.context, loop_info)

    def validate(self):
        loop_info = {'epochs': self.epochs, 'epoch': self.epoch}
        self.callback.on_validation_start(self.context, loop_info)
        self.interaction.validation_start(self.context, loop_info)

        loaders = self.context.valid_loader
        if not isinstance(loaders, (tuple, list)):
            loaders = [loaders]

        summaries = []
        for loader_idx, loader in enumerate(loaders):
            loop_info['batches'] = len(loader)
            loop_info['loader_idx'] = loader_idx

            results = []
            for batch_idx, batch in enumerate(loader):
                loop_info['batch_idx'] = batch_idx
                self.callback.on_validation_batch_start(self.context, batch, loop_info)
                result = self.interaction.validation_step(self.context, batch, loop_info)
                self.callback.on_validation_batch_end(self.context, result, loop_info)
                results.append(result)

            del loop_info['batch_idx']
            summary = self.interaction.validation_summary(self.context, results, loop_info)
            summaries.append(summary)

        del loop_info['batches']
        del loop_info['loader_idx']
        combined_summary = self.interaction.validation_combine_summaries(self.context, summaries, loop_info)
        self.best = self.interaction.get_best(self.context, combined_summary, self.best, loop_info)
        self.callback.on_validation_end(self.context, combined_summary, self.best, loop_info)


class Tester:

    def __init__(self, context: ctx.TestContext, interaction: inter.TestInteraction, callback: clb.Callback) -> None:
        super().__init__()
        self.context = context
        self.interaction = interaction
        self.callback = callback

        self.context.setup()

    def test(self, checkpoint: str):
        self.context.load_checkpoint(checkpoint)

        loop_info = {'batches': len(self.context.test_loader)}
        self.callback.on_test_start(self.context, loop_info)
        self.interaction.test_start(self.context, loop_info)
        results = []
        for batch_idx, batch in enumerate(self.context.test_loader):
            loop_info['batch_idx'] = batch_idx
            self.callback.on_test_batch_start(self.context, batch, loop_info)
            result = self.interaction.test_step(self.context, batch, loop_info)
            self.callback.on_test_batch_end(self.context, result, loop_info)
            results.append(result)

        del loop_info['batch_idx']
        summary = self.interaction.test_summary(self.context, results, loop_info)
        self.callback.on_test_end(self.context, summary, loop_info)
