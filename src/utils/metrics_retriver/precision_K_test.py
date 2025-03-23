
from .Precision_K import Precision_K_articles, Precision_K_rules
import pytest

def test_Precision_K_articles():
    k = 4
    metric = Precision_K_articles(k)
    
    model_output = 'article 111 EPC; rules12a, rules102, guidelines 1'
    ground_truth = 'The board of appeal may either exercise any power within the competence of the department which was responsible for the decision appealed or remit the case to that department for further prosecution (Article 111(1) EPC).'
    
    precision = metric.compute(model_output, ground_truth)
    
    assert precision == 0.25

def test_Precision_K_rules():
    k = 4
    metric = Precision_K_rules(k)
    
    model_output = 'article 11 EPC; rules12a  EPC; rules51 EPC; guidelines 1'
    ground_truth = 'the renewal fee in respect of the third year (only) may be validly paid up to 6 months before it falls due (30 September 2019), Rule 51(1) EPC; Guidelines A-X 5.2.4, OJ EPO 2018 A2 (effective from 01 April 2018).'
    
    precision = metric.compute(model_output, ground_truth)
    
    assert precision == 0.25


def test_Precision_K_articles_2():
    k = 4
    metric = Precision_K_articles(k)
    
    model_output = 'article 111 EPC; rules12a, art75, guidelines 1'
    ground_truth = 'The board arTicles 75(a)(2) of appeal may either exercise any power within the competence of the department which was responsible for the decision appealed or remit the case to that department for further prosecution (Article 111(1) EPC).'
    
    precision = metric.compute(model_output, ground_truth)
    
    assert precision == 0.5


def test_Precision_K_rules_2():
    k = 4
    metric = Precision_K_rules(k)
    
    model_output = 'article 11 EPC; rules12a  EPC; rules51 EPC; guidelines 1'
    ground_truth = 'the renewal fee in respect Rule 12 of the third year (only) may be validly paid up to 6 months before it falls due (30 September 2019), R.51(1) EPC; Guidelines A-X 5.2.4, OJ EPO 2018 A2 (effective from 01 April 2018).'
    
    precision = metric.compute(model_output, ground_truth)
    
    assert precision == 0.5
    
