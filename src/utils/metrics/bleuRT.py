
from ..Metrics import Metric
from evaluate import load

class BLEURT(Metric):

    def __init__(self):
        super().__init__()

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        metric = load("bleurt")
        result = metric.compute(predictions=[model_output], references=[ground_truth])

        if not result or 'scores' not in result :
            return 0.0

        return result['scores'][0]

bleurt = BLEURT()
