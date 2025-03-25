
from evaluate import load
from typing import List
from src.utils.Metrics import Multi_Metric, Adapteur_Multi_2_Metric


class ROUGE(Multi_Metric):

    def __init__(self):
        super().__init__(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        self.metric_eval = load("rouge")

    def _compute_global(self, model_output: str, ground_truth: str) -> List[float]:
        result = self.metric_eval.compute(
            predictions=[model_output], references=[ground_truth])

        if not result:
            return [0., 0., 0., 0.]

        return list(result.values())


__global_rouge = ROUGE()
rouge1 = Adapteur_Multi_2_Metric(__global_rouge, "rouge1")
rouge2 = Adapteur_Multi_2_Metric(__global_rouge, "rouge2")
rougeL = Adapteur_Multi_2_Metric(__global_rouge, "rougeL")
rougeLsum = Adapteur_Multi_2_Metric(__global_rouge, "rougeLsum")
