from src.system.rag import BaseSystem
from src.utils.preprocessing import Dataset
import pandas as pd
import numpy as np

from typing import List, Literal, Dict
from Metrics import Metric


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
