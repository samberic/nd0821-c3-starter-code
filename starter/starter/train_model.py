    # Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from joblib import dump
import pickle
# Add the necessary imports for the starter code.

# Add code to load in the data.
from ml.model import train_model, inference, compute_model_metrics

df = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

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

# Proces the test data with the process_data function.
X_test, y_test, encoder2, lb2 = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f'Precision {precision} recall {recall} fbeta {fbeta}')

with open("../model/encoder", "wb") as f:
    pickle.dump(encoder, f)

with open("../model/lb", "wb") as f:
    pickle.dump(lb, f)

dump(model, '../model/model.joblib')