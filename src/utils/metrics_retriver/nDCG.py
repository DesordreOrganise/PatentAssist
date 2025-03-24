
from ..Metrics import Metric
from ..tools import extract_articles, extract_rules, clean_rule, clean_article
from typing import List, Literal, Dict
from math import log2

def _compute_ndcg(model_rules: List[str], gt_rules: List[str]) -> float:

    if not model_rules:
        return 0.
    
    DCG = 0
    for i in range(len(model_rules)):
        if model_rules[i] in gt_rules:
            DCG += 1 / log2(i + 2)

    IDCG = 0
    for i in range(len(gt_rules)):
        IDCG += 1 / log2(i + 2)

    if IDCG == 0:
        return 1.

    return DCG / IDCG

class NDCG_articles(Metric):

    def __init__(self):
        super().__init__()

    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_split = model_output.split(';') 

        model_output_clean = []
        for output in model_split:
            extracted = extract_articles(output)
            if extracted:
                extracted = extracted[0]
                model_output_clean.append(clean_article(extracted))

            else:
                model_output_clean.append(output)

        
        gt_articles = extract_articles(ground_truth)
        gt_articles = [clean_article(art) for art in gt_articles]
        
        articles_ndcg = _compute_ndcg(model_output_clean, gt_articles)
        
        return articles_ndcg

    def metric_name(self) -> str:
        return f"nDCG_articles"


class NDCG_rules(Metric):

    def __init__(self):
        super().__init__()


    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_split = model_output.split(';') 
        
        model_output_clean = []
        for output in model_split:
            extracted = extract_rules(output)
            if extracted:
                extracted = extracted[0]
                model_output_clean.append(clean_rule(extracted))

            else:
                model_output_clean.append(output)


        gt_rules = extract_rules(ground_truth)
        gt_rules = [clean_rule(rule) for rule in gt_rules]

        rules_ndcg = _compute_ndcg(model_output_clean, gt_rules)
        
        return rules_ndcg
        
    def metric_name(self) -> str:
        return f"nDCG_rules"



class NDCG(Metric):
    
    def __init__(self):
        super().__init__()
        
    
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

        recall = _compute_ndcg(model_output_clean, gt)

        return recall
        


    def metric_name(self) -> str:
        return f"nDCG"
