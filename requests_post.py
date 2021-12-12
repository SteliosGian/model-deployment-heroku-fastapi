import requests
import json


DATA = {
  "age": [
    39
  ],
  "workclass": [
    "State-gov"
  ],
  "fnlgt": [
    77516
  ],
  "education": [
    "Bachelors"
  ],
  "education-num": [
    13
  ],
  "marital-status": [
    "Never-married"
  ],
  "occupation": [
    "Adm-clerical"
  ],
  "relationship": [
    "Not-in-family"
  ],
  "race": [
    "White"
  ],
  "sex": [
    "Male"
  ],
  "capital-gain": [
    2174
  ],
  "capital-loss": [
    0
  ],
  "hours-per-week": [
    40
  ],
  "native-country": [
    "United-States"
  ]
}

URL = "https://udacity-model-deployment.herokuapp.com/predict"
r = requests.post(URL, data=json.dumps(DATA))

print(r.status_code)
print(r.json())
