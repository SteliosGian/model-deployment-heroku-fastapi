# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
from ml.data import process_data
from ml import model

# Add code to load in the data.
import os
print(os.getcwd())
data = pd.read_csv("starter/data/no-space-census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Save encoder and lb
with open("starter/model/encoder", "wb") as file:
    joblib.dump(encoder, file)
with open("starter/model/lb", "wb") as file:
    joblib.dump(lb, file)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train and save a model.
trained_model = model.train_model(X_train, y_train)
joblib.dump(trained_model, "starter/model/trained_model.joblib")

# Get model predictions
preds = model.inference(trained_model, X_test)

# Compute model metrics
precision, recall, fbeta = model.compute_model_metrics(y_test, preds)
print(f"Precision: {round(precision,2)}")
print(f"Recall: {round(recall,2)}")
print(f"Fbeta: {round(fbeta,2)}")

# Output model metrics on sliced data
model.compute_performance_on_slices(trained_model, test, cat_features, encoder, lb)
