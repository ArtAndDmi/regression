from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    add_geometry_features,
    drop_xyz,
    make_cv,
    make_hgbr_pipeline,
    make_transformer,
    replace_zero_dimensions_with_nan,
)


RANDOM_STATE = 42
TARGET = "price"


def apply_selected_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Apply the selected feature engineering from the previous stage.

    Selected experiment:
    geometry_without_xyz_plus_ordinal_clarity
    """

    X = replace_zero_dimensions_with_nan(X)
    X = add_geometry_features(X)
    X = drop_xyz(X)

    return X


def make_selected_pipeline() -> Pipeline:
    """Build the selected pipeline before hyperparameter tuning."""

    return make_hgbr_pipeline(
        transformer=make_transformer(apply_selected_feature_engineering),
        ordinal_clarity=True,
    )


def make_param_grid() -> dict:
    """Create a small and safe grid for HistGradientBoostingRegressor.

    Keep the grid modest because every combination is evaluated with CV.
    With this grid:
    3 * 2 * 2 * 2 = 24 combinations
    24 combinations * 5 folds = 120 model fits
    """

    return {
        "model__learning_rate": [0.04, 0.06, 0.08],
        "model__max_iter": [150, 250],
        "model__max_leaf_nodes": [31, 63],
        "model__min_samples_leaf": [20, 40],
    }


def make_grid_search(
    estimator: Pipeline | None = None,
    param_grid: dict | None = None,
    cv: KFold | None = None,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = 2,
    verbose: int = 1,
) -> GridSearchCV:
    """Create GridSearchCV for the selected pipeline."""

    estimator = estimator or make_selected_pipeline()
    param_grid = param_grid or make_param_grid()
    cv = cv or make_cv()

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        return_train_score=True,
    )


def run_hyperparameter_tuning(
    data: pd.DataFrame,
    target: str = TARGET,
    param_grid: dict | None = None,
    cv: KFold | None = None,
    n_jobs: int = 2,
    verbose: int = 1,
) -> GridSearchCV:
    """Run hyperparameter tuning on train data only."""

    X = data.drop(columns=[target])
    y = data[target]

    search = make_grid_search(
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    search.fit(X, y)

    return search


def get_tuning_results(search: GridSearchCV) -> pd.DataFrame:
    """Convert GridSearchCV results to a clean dataframe."""

    results = pd.DataFrame(search.cv_results_)

    results = results[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "mean_fit_time",
            "params",
        ]
    ].copy()

    results["mean_test_rmse"] = -results["mean_test_score"]
    results["std_test_rmse"] = results["std_test_score"]
    results["mean_train_rmse"] = -results["mean_train_score"]
    results["std_train_rmse"] = results["std_train_score"]

    results = results[
        [
            "rank_test_score",
            "mean_test_rmse",
            "std_test_rmse",
            "mean_train_rmse",
            "std_train_rmse",
            "mean_fit_time",
            "params",
        ]
    ]

    return results.sort_values("rank_test_score")


def get_best_tuning_summary(search: GridSearchCV) -> dict:
    """Return best params and best CV RMSE."""

    return {
        "best_cv_rmse": -search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }


def plot_tuning_results(
    tuning_results: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6),
):
    """Plot top N tuning configurations by CV RMSE."""

    import matplotlib.pyplot as plt

    plot_data = tuning_results.head(top_n).copy()
    plot_data["config"] = [
        f"rank {rank}" for rank in plot_data["rank_test_score"]
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        plot_data["config"],
        plot_data["mean_test_rmse"],
        xerr=plot_data["std_test_rmse"],
    )

    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} hyperparameter configurations")
    ax.set_xlabel("CV RMSE")
    ax.set_ylabel("Configuration")

    for index, value in enumerate(plot_data["mean_test_rmse"]):
        ax.text(
            value,
            index,
            f" {value:.2f}",
            va="center",
        )

    fig.tight_layout()

    return fig, ax


def plot_train_test_gap(
    tuning_results: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6),
):
    """Compare train and validation RMSE for top configurations."""

    import matplotlib.pyplot as plt

    plot_data = tuning_results.head(top_n).copy()
    x = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        x,
        plot_data["mean_train_rmse"],
        marker="o",
        label="train RMSE",
    )
    ax.plot(
        x,
        plot_data["mean_test_rmse"],
        marker="o",
        label="CV RMSE",
    )

    ax.set_title(f"Train vs CV RMSE for top {top_n} configurations")
    ax.set_xlabel("Configuration rank")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["rank_test_score"])
    ax.legend()

    fig.tight_layout()

    return fig, ax