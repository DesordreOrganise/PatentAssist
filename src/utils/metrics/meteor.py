from ..Metrics import Metric
from evaluate import load

class Meteor(Metric):

    def __init__(self):
        super().__init__()

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        metric = load("meteor")
        result = metric.compute(predictions=[model_output], references=[ground_truth])

        if not result or not result['meteor']:
            return 0.0

        return result['meteor']

    def metric_name(self) -> str:
        return "meteor"

meteor_metric = Meteor()


