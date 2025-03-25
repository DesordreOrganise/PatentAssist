
from src.utils.metrics_rag.bleuRT import bleurt
import pytest


def test_bleurt():
    model_output = "hello world"
    ground_truth = "hello world"
    result = bleurt.compute(model_output, ground_truth)
    assert result == 1.0


def test_bleurt_2():
    model_output = "hello world"
    ground_truth = "hello world, how are you"
    result = bleurt.compute(model_output, ground_truth)
    assert result == 0.0
