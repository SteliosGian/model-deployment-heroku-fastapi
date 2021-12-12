from fastapi.testclient import TestClient
import sys
sys.path.insert(0, 'starter')
from main import app

client = TestClient(app)

def test_api_locally_get_root():
    r = client.get('/')
    assert r.status_code == 200, "Status code is not 200"
    assert r.json() == "Welcome, this API returns predictions on Salary", "Wrong json output"

def test_api_locally_post_less_50():
    r = client.post(
        "/predict",
        json={
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
            })
    assert r.status_code == 200, "Status code is not 200"
    assert r.json() == {"Predicted salary": "<=50K"}, "Wrong json output"

def test_api_locally_post_higher_50():
    r = client.post(
        "/predict",
        json={
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
                10000
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
            })
    assert r.status_code == 200, "Status code is not 200"
    assert r.json() == {"Predicted salary": ">50K"}, "Wrong json output"