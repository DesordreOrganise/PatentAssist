from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from typing import List, Literal, Dict


class Metric(ABC):

    def __init__(self):
        self.values : List[float] = []

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
        return float(np.mean(self.values)) if self.values else 0.


class Multi_Metric(ABC):
    
    def __init__(self, metrics_names: List[str]):
        super().__init__()
        self.metrics_names = metrics_names
        self.history = {}

        # to not recompute 'produce' multiple times if not necessary
        self.values_produced = {}
        self.produced_valid = True

    def reset(self):
        self.history = {}

    @abstractmethod
    def _compute_global(self, model_output: str, ground_truth: str) -> List[float]:
        pass


    def _compute_specific(self, model_output: str, ground_truth: str, metric_name: str) -> float:
        couple = model_output + ground_truth
        if couple not in self.history:
            result = self._compute_global(model_output, ground_truth)

            # set the flag to False
            self.produced_valid = False
            
            assert len(result) == len(self.metrics_names)

            history_input = {metric_name: result[i] for i, metric_name in enumerate(self.metrics_names)}
            self.history[couple] = history_input


        assert metric_name in self.metrics_names
        
        return self.history[couple][metric_name]


    def produce(self, metric_name) -> float:
        # if not valid, we recompute the mean for all
        if not self.produced_valid:
            produced_metrics :Dict[str, List[float]] = {metric_name: [] for metric_name in self.metrics_names}

            # fetch all the metrics for each input
            for _, history_input in self.history.items():
                for metric_name in self.metrics_names:
                    produced_metrics[metric_name].append(history_input[metric_name])


            # compute the mean for each metric
            self.values_produced = {metric_name: 0. for metric_name in self.metrics_names}
            for metric_name in self.metrics_names:
                self.values_produced[metric_name] = float(np.mean(produced_metrics[metric_name])) if produced_metrics[metric_name] else 0.
            
            # set the flag to True
            self.produced_valid = True


        assert metric_name in self.metrics_names

        return self.values_produced[metric_name]


class Adapteur_Multi_2_Metric(Metric):
    
    def __init__(self, multi_ref: Multi_Metric, metric_name: str):
        super().__init__()
        self.multi_ref = multi_ref
        self.metric_name = metric_name

        assert metric_name in multi_ref.metrics_names


    def _compute_specific(self, model_output: str, ground_truth: str) -> float:

        # compute the metric for the couple model_output, ground_truth
        output = self.multi_ref._compute_specific(model_output, ground_truth, self.metric_name)
        self.values.append(output)

        return output

    # OVERRIDE
    def produce(self) -> float:
        return self.multi_ref.produce(self.metric_name)



