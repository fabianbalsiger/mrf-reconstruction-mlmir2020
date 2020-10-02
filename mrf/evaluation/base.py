import abc


class BaseEvaluator(abc.ABC):

    @abc.abstractmethod
    def calculate(self) -> dict:
        pass

    @abc.abstractmethod
    def plot(self, root_dir: str):
        pass

    @abc.abstractmethod
    def save(self, root_dir: str):
        pass
