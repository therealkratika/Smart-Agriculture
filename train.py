from pathlib import Path
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path(os.getenv("DATA_PATH", "data/crop_yield.csv"))
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "crop_yield_model.joblib"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")
EXPERIMENT_NAME = "smart-agriculture-yield"
REGISTERED_MODEL = "crop-yield-predictor"

NUMERIC_FEATURES = [
    "nitrogen", "phosphorus", "potassium", "temperature",
    "humidity", "ph", "rainfall"
]
CATEGORICAL_FEATURES = ["crop_type", "soil_type"]
TARGET = "yield_tons_per_hectare"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are required for training.")
    if (df[TARGET] < 0).any():
        raise ValueError("Yield target cannot contain negative values.")
    return df


def build_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES)
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            random_state=42
        )
    }

    best = None
    best_rmse = float("inf")
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            r2 = r2_score(y_test, predictions)
            mlflow.log_params({
                "model_type": name, "test_size": 0.2,
                "random_state": 42, "features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
            })
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )
            print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best = (pipeline, name, mae, rmse, r2, run.info.run_id)

    # Demo quality gate: configurable and deliberately based on RMSE.
    # Set MAX_ACCEPTABLE_RMSE to a domain-appropriate value after validation.
    max_rmse = float(os.getenv("MAX_ACCEPTABLE_RMSE", "999999"))
    if best[3] > max_rmse:
        raise RuntimeError(
            f"Model rejected: RMSE {best[3]:.3f} exceeds threshold {max_rmse:.3f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[0], MODEL_PATH)
    print(f"Best model: {best[1]} | saved to {MODEL_PATH}")

    # Register the winning run and assign champion only after passing the gate.
    model_uri = f"runs:/{best[5]}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    client.set_registered_model_alias(
        name=REGISTERED_MODEL, alias="champion", version=registered.version
    )
    print(f"Registered {REGISTERED_MODEL} version {registered.version} as champion")
    return {"model": best[1], "mae": best[2], "rmse": best[3], "r2": best[4]}


if __name__ == "__main__":
    train()
