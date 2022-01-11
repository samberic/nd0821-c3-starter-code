import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model import train_model


def test_model_type():
    model = train_model(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

    assert isinstance(model, RandomForestClassifier)
