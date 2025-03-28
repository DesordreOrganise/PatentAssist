
from math import log2
from src.utils.metrics_retriever.nDCG import NDCG_articles, NDCG_rules
import pytest


def test_NDCG_4_articles():
    metric = NDCG_articles()

    model_output = 'article 111 ; rules12a; rules102; guidelines 1'
    ground_truth = 'The board of appeal may either exercise any power within the competence of the department which was responsible for the decision appealed or remit the case to that department for further prosecution (Article 111(1) EPC).'

    precision = metric.compute(model_output, ground_truth)

    assert precision == 1


def test_NDCG_4_rules():
    metric = NDCG_rules()

    model_output = 'article 11 ; rules12a  ; rules51 ; guidelines 1'
    ground_truth = 'the renewal fee in respect of the third year (only) may be validly paid up to 6 months before it falls due (30 September 2019), Rule 51(1) EPC; Guidelines A-X 5.2.4, OJ EPO 2018 A2 (effective from 01 April 2018).'

    precision = metric.compute(model_output, ground_truth)

    assert precision == (1 / log2(2+2)) / 1


def test_NDCG_4_articles_2():
    metric = NDCG_articles()

    model_output = 'article 111 ; rules12a; art75; guidelines 1'
    ground_truth = 'The board arTicles 75(a)(2) of appeal may either exercise any power within the competence of the department which was responsible for the decision appealed or remit the case to that department for further prosecution (Article 111(1) EPC).'

    precision = metric.compute(model_output, ground_truth)

    assert precision == (
        ((1 / log2(2 + 0)) + (1 / log2(2+2))) / (1 + (1 / log2(2+1))))


def test_NDGC_K_rules_2():
    metric = NDCG_rules()

    model_output = 'article 11 ; rules12a  ; rules51 ; guidelines 1'
    ground_truth = 'the renewal fee in respect Rule 12 of the third year (only) may be validly paid up to 6 months before it falls due (30 September 2019), R.51(1) EPC; Guidelines A-X 5.2.4, OJ EPO 2018 A2 (effective from 01 April 2018).'

    precision = metric.compute(model_output, ground_truth)

    assert precision == (
        ((1 / log2(2 + 1)) + (1 / log2(2+2))) / (1 + (1 / log2(2+1))))
