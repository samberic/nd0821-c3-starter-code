import pandas as pd
from joblib import load
from model import inference, compute_model_metrics
from data import process_data
import pickle

df = pd.read_csv("../../data/census_clean.csv")
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


def slice_census(df, feature):
    result = ""

    model = load("../../model/model.joblib")

    with open("../../model/encoder", "rb") as enc:
        encoder = pickle.load(enc)
    with open("../../model/lb", "rb") as f:
        lb = pickle.load(f)

    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]

        X_test, y_test, _, _ = process_data(
            df_temp,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        result += f"{feature}: {cls}\n"
        result += f"precision: {precision:.4f}\n"
        result += f"recall: {recall:.4f}\n"
        result += f"fbeta: {fbeta:.4f}\n"

    return result


text = slice_census(df, "education")
text += "========================\n"
text += slice_census(df, "race")

with open("slice_output.txt", "w") as f:
    f.write(text)
