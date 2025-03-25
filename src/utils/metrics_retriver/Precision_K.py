
from src.utils.Metrics import Metric
from src.utils.tools import clean_article, extract_articles, extract_rules, clean_article, clean_rule
from typing import List


class Precision_K_articles(Metric):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_articles = extract_articles(model_output)

        gt_articles = extract_articles(ground_truth)

        articles_precision = self._compute_precision(
            model_articles, gt_articles)

        return articles_precision

    def _compute_precision(self, model_articles: List[str], gt_articles: List[str]) -> float:
        model_articles = model_articles[:self.k]

        # removings spaces and '.' and lowercasing
        model_articles = [article.lower().strip().replace(
            ' ', '').replace('.', '') for article in model_articles]
        # from articles | article to art{number}
        model_articles = [r.replace('s', '').replace(
            'icle', '') for r in model_articles]

        gt_articles = [article.lower().strip().replace(
            ' ', '').replace('.', '') for article in gt_articles]
        gt_articles = [r.replace('s', '').replace('icle', '')
                       for r in gt_articles]

        # print(model_articles)
        # print(gt_articles)

        if not model_articles:
            return 0.

        common_articles = len(
            set(model_articles).intersection(set(gt_articles)))

        if self.k == 0:
            return 1.

        return common_articles / self.k

    def metric_name(self) -> str:
        return f"Precision@{self.k}_articles"


class Precision_K_rules(Metric):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_rules = extract_rules(model_output)

        gt_rules = extract_rules(ground_truth)

        rules_precision = self._compute_precision(model_rules, gt_rules)

        return rules_precision

    def _compute_precision(self, model_rules: List[str], gt_rules: List[str]) -> float:
        model_rules = model_rules[:self.k]

        model_rules = [clean_rule(rule) for rule in model_rules]
        gt_rules = [clean_rule(rule) for rule in gt_rules]

        if not model_rules:
            return 0.

        common_rules = len(set(model_rules).intersection(set(gt_rules)))

        if self.k == 0:
            return 1.

        return common_rules / self.k

    def metric_name(self) -> str:
        return f"Precision@{self.k}_rules"


class Precision_K(Metric):

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

        precision = self._compute_precision(model_output_clean, gt)

        return precision

    def _compute_precision(self, model_output: List[str], gt: List[str]):
        model_output = model_output[:self.k]

        if not model_output:
            return 0.

        common_rules = len(set(model_output).intersection(set(gt)))

        if self.k == 0:
            return 1.

        return common_rules / self.k

    def metric_name(self) -> str:
        return f"Precision@{self.k}"
