"""Evaluate the MLflow Champion and train a separately saved candidate model."""
from pathlib import Path
import argparse, json, os
from datetime import datetime, timezone
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train import (DATA_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
                   build_pipeline, load_data, TRACKING_URI, EXPERIMENT_NAME,
                   REGISTERED_MODEL)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED = FEATURES + [TARGET]
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def load_new_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Incoming dataset is missing columns: {missing}")
    df = df[REQUIRED].copy()
    # Convert numeric fields explicitly; invalid strings become missing and are rejected.
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_counts = df.isna().sum()
    if missing_counts.any():
        raise ValueError(f"Missing/invalid values by column: {missing_counts[missing_counts > 0].to_dict()}")
    if df[CATEGORICAL_FEATURES].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("Incoming data contains blank categorical values.")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Incoming data contains {duplicate_count} duplicate rows.")
    if len(df) < 10:
        raise ValueError("At least 10 labelled incoming rows are required.")
    if (df[TARGET] < 0).any():
        raise ValueError("Yield target cannot contain negative values.")
    print(f"Incoming dataset: {len(df)} rows x {len(df.columns)} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}; duplicate rows: {duplicate_count}")
    print("Target summary:")
    print(df[TARGET].describe().round(3).to_string())
    return df


def metrics(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def load_champion(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@champion")
    return model, str(version.version)


def run(new_data_path="data/new_crop_yield.csv", output_dir="reports", candidate_dir="models/candidate",
        tracking_uri=TRACKING_URI, min_rmse_improvement=0.0):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    old_df = load_data(DATA_PATH)
    new_df = load_new_data(new_data_path)
    champion, champion_version = load_champion(tracking_uri)

    # Reserve part of the incoming labelled batch as a common, untouched evaluation set.
    update_rows, evaluation_rows = train_test_split(new_df, test_size=0.30, random_state=42)
    X_eval, y_eval = evaluation_rows[FEATURES], evaluation_rows[TARGET]
    champion_scores = metrics(y_eval, champion.predict(X_eval))

    # Retrain on all historical training data plus the incoming update portion.
    combined = pd.concat([old_df, update_rows], ignore_index=True).drop_duplicates()
    X_train, y_train = combined[FEATURES], combined[TARGET]
    candidate = build_pipeline(RandomForestRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="monitor-retrain-candidate") as active_run:
        candidate.fit(X_train, y_train)
        candidate_scores = metrics(y_eval, candidate.predict(X_eval))
        mlflow.log_params({
            "workflow": "monitor_retrain", "candidate_model": "RandomForestRegressor",
            "historical_rows": len(old_df), "incoming_rows": len(new_df),
            "incoming_update_rows": len(update_rows), "evaluation_rows": len(evaluation_rows),
            "champion_version": champion_version,
        })
        mlflow.log_metrics({f"champion_{k}": v for k, v in champion_scores.items()})
        mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_scores.items()})
        mlflow.sklearn.log_model(
            candidate,
            name="candidate_model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        run_id = active_run.info.run_id

    # RMSE is primary: lower prediction error in yield units is better.
    # R² is a guardrail to prevent accepting a candidate with worse explained variance.
    rmse_gain = champion_scores["rmse"] - candidate_scores["rmse"]
    eligible = (rmse_gain > min_rmse_improvement and
                candidate_scores["r2"] >= champion_scores["r2"])
    decision = "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_EXISTING_CHAMPION"

    candidate_path = Path(candidate_dir)
    candidate_path.mkdir(parents=True, exist_ok=True)
    artifact_path = candidate_path / "crop_yield_candidate.joblib"
    joblib.dump(candidate, artifact_path)

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "incoming_data": str(new_data_path),
        "incoming_rows": int(len(new_df)),
        "historical_rows": int(len(old_df)),
        "candidate_training_rows_after_deduplication": int(len(combined)),
        "evaluation_rows_from_incoming_batch": int(len(evaluation_rows)),
        "champion_version": champion_version,
        "champion_metrics": champion_scores,
        "candidate_metrics": candidate_scores,
        "quality_gate": {
            "primary_metric": "RMSE (lower is better)",
            "guardrail": "Candidate R2 must be >= Champion R2",
            "minimum_rmse_improvement": min_rmse_improvement,
            "rmse_improvement": float(rmse_gain),
            "decision": decision,
            "champion_alias_changed": False,
        },
        "mlflow_candidate_run_id": run_id,
        "candidate_artifact": str(artifact_path),
        "note": "Candidate is saved separately. This workflow never changes the Champion alias.",
    }
    report_path = report_dir / "monitoring_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    pd.DataFrame([
        {"model": "champion", **champion_scores},
        {"model": "candidate", **candidate_scores},
    ]).to_csv(report_dir / "evaluation_metrics.csv", index=False)
    (report_dir / "quality_gate_decision.json").write_text(json.dumps(report["quality_gate"], indent=2))

    print("\nChampion metrics:", champion_scores)
    print("Candidate metrics:", candidate_scores)
    print(f"Quality gate: {decision}")
    print(f"Candidate artifact: {artifact_path}")
    print(f"Reports: {report_dir.resolve()}")
    print("Champion alias was NOT changed.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-data", default="data/new_crop_yield.csv")
    parser.add_argument("--min-rmse-improvement", type=float, default=0.0)
    args = parser.parse_args()
    run(new_data_path=args.new_data, min_rmse_improvement=args.min_rmse_improvement)
