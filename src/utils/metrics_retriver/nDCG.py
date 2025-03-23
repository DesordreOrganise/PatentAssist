
from ..Metrics import Metric
from ..tools import extract_articles, extract_rules
from typing import List, Literal, Dict
from math import log2

class NDCG_articles(Metric):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_split = model_output.split(';') 

        model_output_clean = []
        for output in model_split:
            extracted = extract_articles(output)
            if extracted:
                extracted = extracted[0]
                extracted = extracted.lower().replace(' ', '').replace('.','')
                extracted = extracted.replace('s', '').replace('icle', '')

                model_output_clean.append(extracted)

            else:
                model_output_clean.append(output)

        
        gt_articles = extract_articles(ground_truth)
        
        articles_ndcg = self._compute_ndcg(model_output_clean, gt_articles)
        
        return articles_ndcg

    def _compute_ndcg(self, model_articles: List[str], gt_articles: List[str]) -> float:
        model_articles = model_articles[:self.k]

        gt_articles = [article.lower().strip().replace(' ', '').replace('.','') for article in gt_articles]
        gt_articles = [r.replace('s', '').replace('icle', '') for r in gt_articles]
        
        if not model_articles:
            return 0.
        

        DCG = 0
        for i in range(min(self.k, len(model_articles))):
            if model_articles[i] in gt_articles:
                DCG += 1 / log2(i + 2)

        IDCG = 0
        for i in range(min(self.k, len(gt_articles))):
            IDCG += 1 / log2(i + 2)

        return DCG / IDCG


class NDCG_rules(Metric):

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
                extracted = extracted.lower().replace(' ', '').replace('.','')
                extracted = extracted.replace('s', '').replace('ule', '')

                model_output_clean.append(extracted)
            else:
                model_output_clean.append(output)


        gt_rules = extract_rules(ground_truth)
        
        rules_ndcg = self._compute_ndcg(model_output_clean, gt_rules)
        
        return rules_ndcg

    def _compute_ndcg(self, model_rules: List[str], gt_rules: List[str]) -> float:
        model_rules = model_rules[:self.k]

        gt_rules = [rule.lower().strip().replace(' ', '').replace('.','') for rule in gt_rules]
        gt_rules = [r.replace('s', '').replace('ule', '') for r in gt_rules]
        

        if not model_rules:
            return 0.
        
        DCG = 0
        for i in range(min(self.k, len(model_rules))):
            if model_rules[i] in gt_rules:
                DCG += 1 / log2(i + 2)

        IDCG = 0
        for i in range(min(self.k, len(gt_rules))):
            IDCG += 1 / log2(i + 2)

        return DCG / IDCG
        
