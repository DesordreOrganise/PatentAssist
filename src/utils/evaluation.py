import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from src.system.rag import BaseSystem
from src.utils.metrics import Metric


class EvaluationFramework():

    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.computed_metrics = pd.Series()


    def evaluate(self, system: BaseSystem, test_data: pd.DataFrame):
        for metric in self.metrics:
            metric.reset()

        iter = 0
        for _, sample in tqdm(test_data.iterrows(), total=len(test_data)):
            iter += 1

            result = system.run(sample["X"])
            for metric in self.metrics:
                metric.compute(result, sample["Y"])

        for metric in self.metrics:
            self.computed_metrics[metric.metric_name()] = metric.produce()

        print(self.computed_metrics)
        return self.computed_metrics
