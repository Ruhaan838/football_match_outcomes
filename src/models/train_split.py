from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.data.process_data import process_football_data
from config import APIkey


def get_data(split_type: str = "regression", test_size=0.2, verbose=False):
    data = process_football_data(APIkey.api_key, verbose=verbose)

    feature_cols = [
        "match_day", "match_month", "match_weekday", "match_year",
        "league.name", "league.country", "league.round",
        "fixture.venue.name", "fixture.venue.city",
        "teams.home.name", "teams.away.name",
    ]

    if split_type == "regression":
        target_cols = ["goals.home", "goals.away"]
        is_classification = False
    elif split_type == "b_classification":
        target_cols = ["match_result"]
        is_classification = True
    else:
        raise ValueError(f"Invalid split_type: {split_type}")

    X = data[feature_cols]
    y = data[target_cols]

    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=42
    )

    label_encoder = None

    if is_classification:
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train.values.ravel())

        y_test_flat = y_test.values.ravel()
        mask = pd.Series(y_test_flat).isin(label_encoder.classes_)
        X_test = X_test[mask.values]
        y_test = y_test_flat[mask.values]

        y_test_enc = label_encoder.transform(y_test)
        y_train, y_test = y_train_enc, y_test_enc

    return X_train, X_test, y_train, y_test, label_encoder
