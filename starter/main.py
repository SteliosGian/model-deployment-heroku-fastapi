# Put the code for your API here.
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Callable, List
import sys
sys.path.insert(0, 'starter/starter')
from starter.ml import model, data

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Load model and encoders
TRAINED_MODEL = joblib.load("starter/model/trained_model.joblib")
ENCODER = joblib.load("starter/model/encoder")
LB = joblib.load("starter/model/lb")

class InferenceRequest(BaseModel):
    age: List[int]
    workclass: List[str]
    fnlgt: List[int]
    education: List[str]
    education_num: List[int] = Field(alias='education-num')
    marital_status: List[str] = Field(alias='marital-status')
    occupation: List[str]
    relationship: List[str]
    race: List[str]
    sex: List[str]
    capital_gain: List[int] = Field(alias='capital-gain')
    capital_loss: List[int] = Field(alias='capital-loss')
    hours_per_week: List[int] = Field(alias='hours-per-week')
    native_country: List[str] = Field(alias='native-country')


@app.get('/')
async def welcome():
    return "Welcome, this API returns predictions on Salary"


@app.post("/predict")
async def predict(request: InferenceRequest = Body(
    ...,
    example={
        "age": [39],
        "workclass": ["State-gov"],
        "fnlgt": [77516],
        "education": ["Bachelors"],
        "education-num": [13],
        "marital-status": ["Never-married"],
        "occupation": ["Adm-clerical"],
        "relationship": ["Not-in-family"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [2174],
        "capital-loss": [0],
        "hours-per-week": [40],
        "native-country": ["United-States"]
    }
)):
    data_alias = jsonable_encoder(request, by_alias=True)
    to_predict = pd.DataFrame.from_dict(data_alias)
    processed_data, _, _, _ = data.process_data(to_predict, categorical_features=CATEGORICAL_FEATURES, label=None, training=False, encoder=ENCODER, lb=LB)
    preds = model.inference(model=TRAINED_MODEL, X=np.array(processed_data))

    return {"Predicted salary": preds[0]}
