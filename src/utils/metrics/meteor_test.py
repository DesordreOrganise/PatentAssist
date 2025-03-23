
from .meteor import Meteor
import pytest

def test_meteor():
    model_output = "hello world"
    ground_truth = "hello world"
    result = Meteor().compute(model_output, ground_truth)
    assert result == 1.0

def test_meteor_2():
    model_output = "hello world"
    ground_truth = "hello world, how are you"
    result = Meteor().compute(model_output, ground_truth)
    assert result == 0.0
