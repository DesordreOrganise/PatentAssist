
from src.utils.metrics import Metric
from src.utils.tools import extract_articles, extract_rules, clean_article, clean_rule
from typing import List


def _compute_recall(model_output: List[str], gt: List[str], k: int):
    model_output = model_output[:k]

    if len(gt) == 0:
        return 1.

    if not model_output:
        return 0.

    common_files = len(set(model_output).intersection(set(gt)))

    return common_files / len(gt)


class Recall_K_articles(Metric):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_articles = extract_articles(model_output)
        model_articles = [clean_article(art) for art in model_articles]

        gt_articles = extract_articles(ground_truth)
        gt_articles = [clean_article(art) for art in gt_articles]

        articles_recall = _compute_recall(model_articles, gt_articles, self.k)

        return articles_recall

    def metric_name(self) -> str:
        return f"Recall@{self.k}_articles"


class Recall_K_rules(Metric):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_rules = extract_rules(model_output)
        model_rules = [clean_rule(rule) for rule in model_rules]

        gt_rules = extract_rules(ground_truth)
        gt_rules = [clean_rule(rule) for rule in gt_rules]

        rules_recall = _compute_recall(model_rules, gt_rules, self.k)

        return rules_recall

    def metric_name(self) -> str:
        return f"Recall@{self.k}_rules"


class Recall_K(Metric):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_split = model_output.split(';')
        model_output_clean = []

        for output in model_split:
            extracted = extract_rules(output)
            if extracted:
                extracted = extracted[0]
                model_output_clean.append(clean_rule(extracted))

                continue

            extracted = extract_articles(output)
            if extracted:
                extracted = extracted[0]
                model_output_clean.append(clean_article(extracted))

                continue

            model_output_clean.append(output)

        gt_articles = extract_articles(ground_truth)
        gt_rules = extract_rules(ground_truth)

        gt = [clean_rule(rule) for rule in gt_rules]
        gt.extend([clean_article(article) for article in gt_articles])

        recall = _compute_recall(model_output_clean, gt, self.k)

        return recall

    def metric_name(self) -> str:
        return f"Recall@{self.k}"
