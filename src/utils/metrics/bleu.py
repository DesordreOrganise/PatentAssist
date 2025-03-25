
from src.utils.Metrics import Metric
from evaluate import load


class BLEU(Metric):

    def __init__(self):
        super().__init__()
        self.metric_eval = load("bleu")

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        result = self.metric_eval.compute(
            predictions=[model_output], references=[ground_truth])

        if not result or 'bleu' not in result:
            return 0.0

        return result['bleu']

    def metric_name(self) -> str:
        return "bleu"


bleu = BLEU()
