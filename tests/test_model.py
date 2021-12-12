import pytest
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.ml import model

@pytest.fixture
def data():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "numerical_feat": [3.0, 15.0, 25.9],
        "target_feat": ["yes", "no", "no"]
    })
    return df


def test_train_model(data):
    X = data[['numerical_feat']]
    y = data['target_feat']
    trained_model = model.train_model(X, y)
    fake_model = RandomForestClassifier()
    fake_model_ = fake_model.fit(X, y)
    assert type(trained_model) == type(fake_model_), "Fake model and trained model are not of the same type"
    assert trained_model is not fake_model_, "Trained model is same instance as fake model"


def test_compute_model_metrics():
    y = [1, 0, 0]
    y_preds = [1, 1, 0]
    precision, recall, fbeta = model.compute_model_metrics(y, y_preds)
    assert precision == 0.5, "Precision incorrect" # precision = True pos / (True pos + False pos)
    assert recall == 1.0, "Recall incorrect" # recall = True pos / (True pos + False neg)
    assert math.isclose(fbeta, 0.6666, rel_tol=1e-04), "fbeta incorrect" # fbeta = (1 + beta^2) * ((precision * recall) / (beta^2 * precision + recall))



def test_inference(data):
    X = data[['numerical_feat']]
    y = data['target_feat']
    fake_model = RandomForestClassifier()
    fake_model_ = fake_model.fit(X, y)
    preds = model.inference(fake_model_, X)
    assert type(preds) == np.ndarray, "Output is not numpy.ndarray"
    assert len(preds) == X.shape[0], "Output has wrong shape"
