
from src.utils.metrics import Metric
from evaluate import load


class BLEURT(Metric):

    def __init__(self):
        super().__init__()
        self.metric_eval = load("bleurt")

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        result = self.metric_eval.compute(
            predictions=[model_output], references=[ground_truth])

        if not result or 'scores' not in result:
            return 0.0

        return result['scores'][0]

    def metric_name(self) -> str:
        return "bleurt"


bleurt = BLEURT()
