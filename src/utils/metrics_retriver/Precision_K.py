
from ..Metrics import Metric
from ..tools import extract_articles, extract_rules
from typing import List, Literal, Dict

class Precision_K_articles(Metric):
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    
    def _compute_specific(self, model_output: str, ground_truth: str) -> float:
        model_articles = extract_articles(model_output)
        
        gt_articles = extract_articles(ground_truth)
        
        articles_precision = self._compute_precision(model_articles, gt_articles)
        
        return articles_precision 
    
    def _compute_precision(self, model_articles: List[str], gt_articles: List[str]) -> float:
        model_articles = model_articles[:self.k]
        
        #removings spaces and '.' and lowercasing
        model_articles = [article.lower().strip().replace(' ', '').replace('.','') for article in model_articles]
        #from articles | article to art{number}
        model_articles = [r.replace('s', '').replace('icle', '') for r in model_articles]

        
        gt_articles = [article.lower().strip().replace(' ', '').replace('.','') for article in gt_articles]
        gt_articles = [r.replace('s', '').replace('icle', '') for r in gt_articles]

        print(model_articles)
        print(gt_articles)

        if not model_articles:
            return 0.
        
        common_articles = len(set(model_articles).intersection(set(gt_articles)))
        
        return common_articles / self.k

        
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

        #removings spaces and '.' and lowercasing
        model_rules = [article.lower().replace(' ', '').replace('.', '') for article in model_rules if article]
        #from rules | rules to r{number}
        model_rules = [r.replace('s', '').replace('ule', '') for r in model_rules]

        gt_rules = [article.lower().replace(' ', '').replace('.', '') for article in gt_rules if article]
        gt_rules = [r.replace('s', '').replace('ule', '') for r in gt_rules]
        
        if not model_rules:
            return 0.
        
        common_rules = len(set(model_rules).intersection(set(gt_rules)))
        
        return common_rules / self.k


