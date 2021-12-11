import json
from ml.data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
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


def compute_model_metrics(y, preds):
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


def inference(model, X):
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
    return preds


def compute_performance_on_slices(model, test_data, cat_features, encoder, lb, ):
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
