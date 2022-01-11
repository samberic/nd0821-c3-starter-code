import requests

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
print(f'status code: {resp.status_code}')
print(f'results: {resp.json()}')