from src.utils.Metrics import Metric
from evaluate import load


class Meteor(Metric):

    def __init__(self):
        super().__init__()
        self.metric_eval = load("meteor")

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        result = self.metric_eval.compute(
            predictions=[model_output], references=[ground_truth])

        if not result or not result['meteor']:
            return 0.0

        return result['meteor']

    def metric_name(self) -> str:
        return "meteor"


meteor_metric = Meteor()
