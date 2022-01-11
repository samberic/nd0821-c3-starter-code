import pickle
import uvicorn

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

# Instantiate the app.
from starter.ml.data import process_data

app = FastAPI()


def to_dashes(string: str) -> str:
    return string.replace("_", "-")


class Value(BaseModel):
    age: int
    workclass: str
    fnlgt: str
    education: str
    education_num: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = to_dashes
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": "77516",
                "education": "Bachelors",
                "education_num": "13",
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-In-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to this exciting coursework!"}


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

model = load("model/model.joblib")

with open("model/encoder", "rb") as enc:
    encoder = pickle.load(enc)
with open("model/lb", "rb") as f:
    lb = pickle.load(f)


@app.post("/model/")
async def do_inference(item: Value):
    d = item.dict(by_alias=True)
    test_frame = pd.DataFrame([d], columns=d.keys())
    X, _, _, _ = process_data(
        test_frame,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y = model.predict(X)
    return {"result": lb.inverse_transform(y)[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
