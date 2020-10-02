import abc

import torch


class TestContext:

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.test_loader = None

    def setup(self):
        self.model = self._init_model()
        self.test_loader = self._init_test_loader()

    def _init_model(self):
        raise NotImplementedError()

    def _init_test_loader(self):
        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint.pop('model_state_dict'))


class Context(TestContext, abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None

    def setup(self):
        super().setup()
        self.optimizer = self._init_optimizer()
        self.train_loader = self._init_train_loader()
        self.valid_loader = self._init_valid_loader()
        self._setup_additional()

    def _setup_additional(self):
        pass

    def _init_optimizer(self):
        raise NotImplementedError()

    def _init_train_loader(self):
        raise NotImplementedError()

    def _init_valid_loader(self):
        return None

    def _init_test_loader(self):
        return None

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint.pop('model_state_dict'))
        self.optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
        epoch = checkpoint.pop('epoch')
        best = checkpoint.pop('best')
        self._load_additional(checkpoint)
        return epoch, best

    def _load_additional(self, checkpoint: dict):
        pass

    def save_checkpoint(self, checkpoint_path, epoch: int, best: dict):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best': best
        }
        checkpoint.update(self._save_additional())
        torch.save(checkpoint, checkpoint_path)

    def _save_additional(self):
        return {}
