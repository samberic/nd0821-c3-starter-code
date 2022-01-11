import requests


def test_GET():
    resp = requests.get("https://sam-course-ml-app.herokuapp.com/")
    assert (resp.status_code == 200)
    assert (resp.json() == {
        "greeting": "Welcome to this exciting coursework!"
    })


def test_POST():
    resp = requests.post("https://sam-course-ml-app.herokuapp.com/model",
                         json={
                             "age": 39,
                             "workclass": "State-gov",
                             "fnlgt": "77516",
                             "education": "Bachelors",
                             "education-num": "13",
                             "marital-status": "Never-married",
                             "occupation": "Adm-clerical",
                             "relationship": "Not-In-family",
                             "race": "White",
                             "sex": "Male",
                             "capital-gain": 2174,
                             "capital-loss": 0,
                             "hours-per-week": 40,
                             "native-country": "United-States"
                         })
    assert (resp.status_code == 200)
    assert (resp.json() == {
        "result": "<=50K"
    })


def test_inference_over_50K():
    resp = requests.post("https://sam-course-ml-app.herokuapp.com/model",
                         json={
                             "age": 52,
                             "workclass": "Self-emp-not-inc",
                             "fnlgt": "209642",
                             "education": "HS-grad",
                             "education-num": "9",
                             "marital-status": "Married-civ-spouse",
                             "occupation": "Exec-managerial",
                             "relationship": "Husband",
                             "race": "White",
                             "sex": "Male",
                             "capital-gain": 0,
                             "capital-loss": 0,
                             "hours-per-week": 45,
                             "native-country": "United-States"
                         })
    assert (resp.status_code == 200)
    assert (resp.json() == {
        "result": ">50K"
    })
