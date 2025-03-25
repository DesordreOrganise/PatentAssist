
from src.utils.metrics.bert_score import bert_score_acc, bert_score_recall, bert_score_f1
import pytest


def test_bert_score_acc():
    model_output = "hello world"
    ground_truth = "hello world"
    result = bert_score_acc.compute(model_output, ground_truth)
    assert result == 1.0


def test_meteor_recall():
    model_output = "hello world"
    ground_truth = "hello world, how are you"
    result = bert_score_recall.compute(model_output, ground_truth)
    assert result == 0.0


def test_bert_score_f1():
    model_output = "hello world"
    ground_truth = "hello world"
    result = bert_score_f1.compute(model_output, ground_truth)
    assert result == 1.0
