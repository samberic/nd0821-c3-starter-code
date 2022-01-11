import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model import train_model, inference


def test_model_type():
    model = train_model(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

    assert isinstance(model, RandomForestClassifier)


def test_inference_type():
    model = train_model(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
    prediction = inference(model, [[1, 1]])
    assert isinstance(prediction, np.ndarray)

def test_inference_result():
    model = train_model(np.array([[1, 1], [3, 4]]), np.array([1, 2]))
    prediction = inference(model, [[1, 1]])
    assert prediction == [1]



