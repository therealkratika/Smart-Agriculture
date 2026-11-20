import pandas as pd
import pytest
from train import load_data, build_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from monitor_retrain import SKOPS_TRUSTED_TYPES
from sklearn.ensemble import RandomForestRegressor


def test_load_data_has_expected_columns():
    df = load_data()
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).issubset(df.columns)
    assert len(df) >= 30


def test_pipeline_can_fit_and_predict():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    model = build_pipeline(RandomForestRegressor(n_estimators=10, random_state=42))
    model.fit(X.iloc[:30], y.iloc[:30])
    prediction = model.predict(X.iloc[30:32])
    assert len(prediction) == 2
    assert all(value >= 0 for value in prediction)


def test_load_data_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("data/does-not-exist.csv")


def test_monitor_retrain_uses_explicit_skops_trust_allowlist():
    assert "numpy.dtype" in SKOPS_TRUSTED_TYPES
    assert "sklearn.tree._tree.Tree" in SKOPS_TRUSTED_TYPES
