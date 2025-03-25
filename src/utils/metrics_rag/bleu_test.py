from src.utils.metrics_rag.bleu import bleu
import pytest


def test_bleu():
    model_output = "hello world, how are you"
    ground_truth = "hello world, how are you"
    result = bleu.compute(model_output, ground_truth)
    assert result == 1.0


def test_bleu_2():
    model_output = "hello there generol kenobi"
    ground_truth = "hello there, general kenobi"
    result = bleu.compute(model_output, ground_truth)

    assert result == 0.0
