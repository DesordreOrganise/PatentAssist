# file to test the rouge score calculation
from src.utils.metrics_rag.rouge import rouge1, rouge2, rougeL, rougeLsum
import pytest


def test_rouge1():
    model_output = "hello world"
    ground_truth = "hello world"
    result = rouge1.compute(model_output, ground_truth)
    assert result == 1.0


def test_rouge2():
    model_output = "hello world"
    ground_truth = "hello world"
    result = rouge2.compute(model_output, ground_truth)
    assert result == 1.0


def test_rougeL():
    model_output = "hello world"
    ground_truth = "hello world"
    result = rougeL.compute(model_output, ground_truth)
    assert result == 1.0


def test_rougeLsum():
    model_output = "hello world"
    ground_truth = "hello world"
    result = rougeLsum.compute(model_output, ground_truth)
    assert result == 1.0


def test_rouge1_2():
    model_output = "hella world"
    ground_truth = "hello world"
    result = rouge1.compute(model_output, ground_truth)
    assert result == 0.5


def test_rouge2_2():
    model_output = "hella world"
    ground_truth = "hello world"
    result = rouge2.compute(model_output, ground_truth)
    assert result == 0.
