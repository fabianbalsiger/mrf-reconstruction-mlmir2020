import torch

import mrf.loop.context as ctx


class TestInteraction:

    def test_start(self, context: ctx.TestContext, loop_info):
        torch.set_grad_enabled(False)
        context.model.eval()

    def test_step(self, context: ctx.TestContext, batch, loop_info):
        raise NotImplementedError()

    def test_summary(self, context: ctx.TestContext, results, loop_info):
        return {}


class Interaction(TestInteraction):

    def epoch_start(self, context: ctx.Context, loop_info):
        torch.set_grad_enabled(True)
        context.model.train()

    def training_step(self, context: ctx.Context, batch, loop_info):
        raise NotImplementedError()

    def train_summary(self, context: ctx.Context, results, loop_info):
        return {}

    def validation_start(self, context: ctx.Context, loop_info):
        torch.set_grad_enabled(False)
        context.model.eval()

    def validation_step(self, context: ctx.Context, batch, loop_info):
        return {}

    def validation_summary(self, context: ctx.Context, results, loop_info):
        return {}

    def validation_combine_summaries(self, context: ctx.Context, summaries, loop_info):
        return {k:v for summary in summaries for k, v in summary.items()}

    def get_best(self, context: ctx.Context, summary, best, loop_info):
        return best

    def test_step(self, context: ctx.TestContext, batch, loop_info):
        return {}
