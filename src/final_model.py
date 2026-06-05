"""Final model utilities for the Diamonds regression project."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    add_geometry_features,
    drop_xyz,
    make_hgbr_pipeline,
    make_transformer,
    replace_zero_dimensions_with_nan,
)


RANDOM_STATE = 42
TARGET = "price"


def apply_final_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Apply selected feature engineering from the regression project."""

    X = replace_zero_dimensions_with_nan(X)
    X = add_geometry_features(X)
    X = drop_xyz(X)

    return X


def make_final_model() -> Pipeline:
    """Create final tuned regression pipeline."""

    model = make_hgbr_pipeline(
        transformer=make_transformer(apply_final_feature_engineering),
        ordinal_clarity=True,
    )

    model.set_params(
        model__learning_rate=0.04,
        model__max_iter=250,
        model__max_leaf_nodes=63,
        model__min_samples_leaf=20,
    )

    return model


def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit final model on the full train data."""

    model = make_final_model()
    model.fit(X_train, y_train)

    return model


def predict_final_model(
    model: Pipeline,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Predict target values for test data."""

    return model.predict(X_test)


def evaluate_regression_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate final regression metrics."""


    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def make_prediction_results(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Create dataframe with actual values, predictions and residuals."""

    results = X_test.copy()

    results["actual_price"] = y_test.values
    results["predicted_price"] = y_pred
    results["residual"] = results["actual_price"] - results["predicted_price"]
    results["absolute_error"] = results["residual"].abs()

    return results


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple[int, int] = (7, 7),
):
    """Plot actual vs predicted target values."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.4,
    )

    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        color="red",
    )

    ax.set_title("Actual vs predicted price")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")

    fig.tight_layout()

    return fig, ax


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple[int, int] = (10, 5),
):
    """Plot residuals against predicted values."""

    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true) - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.4,
    )

    ax.axhline(
        0,
        linestyle="--",
        color="red",
    )

    ax.set_title("Residuals vs predicted price")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual")

    fig.tight_layout()

    return fig, ax


def plot_residual_distribution(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    bins: int = 50,
    figsize: tuple[int, int] = (10, 5),
):
    """Plot residual distribution."""

    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true) - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        residuals,
        bins=bins,
    )

    ax.axvline(
        0,
        linestyle="--",
        color="red",
    )

    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")

    fig.tight_layout()

    return fig, ax


def get_largest_errors(
    prediction_results: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return rows with the largest absolute prediction errors."""

    return prediction_results.sort_values(
        "absolute_error",
        ascending=False,
    ).head(top_n)