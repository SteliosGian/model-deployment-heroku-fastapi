import json
import sys
sys.path.insert(0, 'starter/starter')
import numpy as np
import pandas as pd
from ml.data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.array, y_train: np.array):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    return random_forest


def compute_model_metrics(y: np.array, preds: np.array):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X: np.array) -> np.array:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    preds = np.where(preds == 0, '<=50K', '>50K')
    return preds


def compute_performance_on_slices(model, test_data: pd.DataFrame, cat_features: list, encoder, lb) -> dict:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest Classifier
        Trained machine learning model.
    test_data : pd.DataFrame
        Data used for prediction.
    cat_features : list
        List of categorical features.
    encoder : joblib file
        One Hot encoder
    lb : joblib file
        Label encoder
    Returns
    -------
    model_metrics_dict : dict
        Dictionary with model metrics per slice.
    """
    model_metrics_dict = {}
    
    for slice in test_data['education'].unique():
        sliced_data= test_data[test_data['education'] == slice]
        sliced_X, sliced_y, _, _ = process_data(sliced_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

        sliced_preds = model.predict(sliced_X)
        precision, recall, fbeta = compute_model_metrics(sliced_y, sliced_preds)
        model_metrics_dict[slice] = {"precision": precision, "recall": recall, "fbeta": fbeta}

    with open("starter/model/slice_output.txt", "w") as file:
        file.write(json.dumps(model_metrics_dict))
    
    return model_metrics_dict
