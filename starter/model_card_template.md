# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Stelios Giannikis created the model. It is a Random Forest model using the default hyperparameters in scikit-learn 1.0.1
## Intended Use
The model should be used to predict the salary of a person based off specific attributed related to that person.
## Training Data
The dataset was acquired from the UCI repository (https://archive.ics.uci.edu/ml/datasets/census+income). The training set is 80% of the original dataset by performing a random split.
## Evaluation Data
The evaluation set is 20% of the original dataset by performing a random split.
## Metrics
The model was evaluated using Precision, Recall and Fbeta score. The values were respectively 0.75, 0.63, and 0.69.
## Ethical Considerations
This model and dataset should not be used to discriminate on employees.
## Caveats and Recommendations
The performance of this model is acquired without performing any hyperparameter tuning. This means that there is potential to increase the model performance by adjusting its hyperparameters. Moreover, other ML models were not considered for this project so there could be also space for improvement by analyzing the performance of other ML models such as Boosting, Logistic Regression or Neural Networks.
