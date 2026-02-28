from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_FEATURES = [
    "cement",
    "water_cement_ratio",
    "admixture",
    "temperature",
    "humidity",
    "time_hours",
]


def _encode_features(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    encoded = pd.get_dummies(frame[BASE_FEATURES + ["curing_type"]], columns=["curing_type"])

    if feature_columns is None:
        return encoded

    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    return aligned


def train_strength_model(data_path: str = "data/sample_production_data.csv", random_state: int = 42) -> dict:
    csv_path = Path(data_path)
    data = pd.read_csv(csv_path)

    x = _encode_features(data)
    y = data["strength"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)

    return {
        "model": model,
        "feature_columns": x.columns.tolist(),
        "curing_types": sorted(data["curing_type"].unique().tolist()),
        "train_score": model.score(x_train, y_train),
        "test_score": model.score(x_test, y_test),
    }


def predict_strength(
    bundle: dict,
    cement: float,
    water_cement_ratio: float,
    admixture: float,
    temperature: float,
    humidity: float,
    curing_type: str,
    time_hours: float,
) -> float:
    row = pd.DataFrame(
        [
            {
                "cement": cement,
                "water_cement_ratio": water_cement_ratio,
                "admixture": admixture,
                "temperature": temperature,
                "humidity": humidity,
                "curing_type": curing_type,
                "time_hours": time_hours,
            }
        ]
    )

    x_row = _encode_features(row, bundle["feature_columns"])
    return float(bundle["model"].predict(x_row)[0])


def predict_strength_curve(bundle: dict, fixed_params: dict, curing_type: str, time_points) -> list[float]:
    strengths = []

    for time_hours in time_points:
        strengths.append(
            predict_strength(
                bundle=bundle,
                cement=fixed_params["cement"],
                water_cement_ratio=fixed_params["water_cement_ratio"],
                admixture=fixed_params["admixture"],
                temperature=fixed_params["temperature"],
                humidity=fixed_params["humidity"],
                curing_type=curing_type,
                time_hours=float(time_hours),
            )
        )

    return strengths
