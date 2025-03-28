
from src.utils.metrics import Multi_Metric, Adapteur_Multi_2_Metric
from typing import List
import torch
from evaluate import load


class BertScore(Multi_Metric):

    def __init__(self):
        super().__init__(["precision", "recall", "f1"])
        self.metric_eval = load("bertscore")

        # Forcer l'utilisation du CPU
        # self.device = torch.device("cpu")
        # self.metric_eval.model.to(self.device)


    def _compute_global(self, model_output: str, ground_truth: str) -> List[float]:
        result = self.metric_eval.compute(predictions=[model_output], references=[
            ground_truth], model_type="distilbert-base-uncased", device='cpu')

        if not result or not result['precision'] or not result['recall'] or not result['f1']:
            return [0., 0., 0.]
        return [result['precision'], result['recall'], result['f1']]


bert_score = BertScore()
bert_score_acc = Adapteur_Multi_2_Metric(bert_score, "precision")
bert_score_recall = Adapteur_Multi_2_Metric(bert_score, "recall")
bert_score_f1 = Adapteur_Multi_2_Metric(bert_score, "f1")
