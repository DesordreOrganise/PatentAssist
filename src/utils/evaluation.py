from abc import ABC, abstractmethod
from src.system.rag import BaseSystem
from src.utils.preprocessing import Dataset
import pandas as pd
import numpy as np


class Metric(ABC):

    def __init__(self):
        self.values = []

    def reset(self):
        self.values = []

    @abstractmethod
    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        pass


    def compute(self, model_output: str, ground_truth: str):
        result = self._compute_specific(model_output, ground_truth)
        self.values.append(result)
        return result


    def produce(self) -> float:
        return np.mean(self.values) if self.values else 0


class EvaluationFramework():

    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.computed_metrics = pd.Series()

    def evaluate(self, system: BaseSystem, test_data: pd.DataFrame):
        for metric in self.metrics:
            metric.reset()
        for _, sample in test_data.iterrows():
            result = system.run(sample["input"])
            for metric in self.metrics:
                metric.compute(result, sample["gold"])