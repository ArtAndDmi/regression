from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


RANDOM_STATE = 42

TARGET = "price"

CATEGORICAL_FEATURES = ["cut", "color", "clarity"]

CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


@dataclass
class FeatureEngineeringExperiment:
    name: str
    description: str
    estimator: object


def make_cv() -> KFold:
    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_geometry_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X["volume"] = X["x"] * X["y"] * X["z"]
    X["base_area"] = X["x"] * X["y"]
    X["xy_ratio"] = X["x"] / X["y"].replace(0, np.nan)

    return X


def add_log_carat(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["log1p_carat"] = np.log1p(X["carat"])
    return X


def replace_zero_dimensions_with_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    zero_dimensions_mask = (X[["x", "y", "z"]] == 0).all(axis=1)
    X.loc[zero_dimensions_mask, ["x", "y", "z"]] = np.nan

    return X


def drop_xyz(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    return X.drop(columns=["x", "y", "z"])


def make_preprocessor(
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ordinal_clarity: bool = False,
) -> ColumnTransformer:
    categorical_features = list(categorical_features)

    if ordinal_clarity:
        one_hot_features = [
            feature for feature in categorical_features
            if feature != "clarity"
        ]

        return ColumnTransformer(
            transformers=[
                (
                    "onehot",
                    make_one_hot_encoder(),
                    one_hot_features,
                ),
                (
                    "clarity_ordinal",
                    OrdinalEncoder(
                        categories=[CLARITY_ORDER],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ["clarity"],
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                make_one_hot_encoder(),
                categorical_features,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def make_hgbr_pipeline(
    transformer: FunctionTransformer | None = None,
    ordinal_clarity: bool = False,
) -> Pipeline:
    steps = []

    if transformer is not None:
        steps.append(("feature_engineering", transformer))

    steps.extend(
        [
            (
                "preprocessing",
                make_preprocessor(ordinal_clarity=ordinal_clarity),
            ),
            (
                "model",
                HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            ),
        ]
    )

    return Pipeline(steps)


def make_log_target_pipeline(
    transformer: FunctionTransformer | None = None,
    ordinal_clarity: bool = False,
) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=make_hgbr_pipeline(
            transformer=transformer,
            ordinal_clarity=ordinal_clarity,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_transformer(function: Callable[[pd.DataFrame], pd.DataFrame]) -> FunctionTransformer:
    return FunctionTransformer(
        func=function,
        validate=False,
    )


def make_feature_engineering_experiments() -> list[FeatureEngineeringExperiment]:
    return [
        FeatureEngineeringExperiment(
            name="baseline_hgbr",
            description="Baseline HistGradientBoostingRegressor without feature engineering.",
            estimator=make_hgbr_pipeline(),
        ),
        FeatureEngineeringExperiment(
            name="zero_dimensions_to_nan",
            description="Replace rows where x, y, z are all zero with NaN.",
            estimator=make_hgbr_pipeline(
                transformer=make_transformer(replace_zero_dimensions_with_nan),
            ),
        ),
        FeatureEngineeringExperiment(
            name="geometry_features",
            description="Add volume, base_area and xy_ratio.",
            estimator=make_hgbr_pipeline(
                transformer=make_transformer(add_geometry_features),
            ),
        ),
        FeatureEngineeringExperiment(
            name="zero_dimensions_plus_geometry",
            description="Replace zero dimensions with NaN, then add geometry features.",
            estimator=make_hgbr_pipeline(
                transformer=make_transformer(
                    lambda X: add_geometry_features(
                        replace_zero_dimensions_with_nan(X)
                    )
                ),
            ),
        ),
        FeatureEngineeringExperiment(
            name="log_carat",
            description="Add log1p(carat) as an additional feature.",
            estimator=make_hgbr_pipeline(
                transformer=make_transformer(add_log_carat),
            ),
        ),
        FeatureEngineeringExperiment(
            name="ordinal_clarity",
            description="Use ordinal encoding for clarity instead of one-hot encoding.",
            estimator=make_hgbr_pipeline(
                ordinal_clarity=True,
            ),
        ),
        FeatureEngineeringExperiment(
            name="geometry_without_xyz",
            description="Use geometry features and remove raw x, y, z.",
            estimator=make_hgbr_pipeline(
                transformer=make_transformer(
                    lambda X: drop_xyz(
                        add_geometry_features(
                            replace_zero_dimensions_with_nan(X)
                        )
                    )
                ),
            ),
        ),
        FeatureEngineeringExperiment(
            name="geometry_without_xyz_plus_ordinal_clarity",
            description="Use geometry features without raw x, y, z and ordinal encoding for clarity.",
            estimator=make_hgbr_pipeline(
                transformer=make_transformer(
                    lambda X: drop_xyz(
                        add_geometry_features(
                            replace_zero_dimensions_with_nan(X)
                        )
                    )
                ),
                ordinal_clarity=True,
            ),
        ),
        FeatureEngineeringExperiment(
            name="log_target",
            description="Train model on log1p(price) using TransformedTargetRegressor.",
            estimator=make_log_target_pipeline(),
        ),
    ]


def run_feature_engineering_experiments(
    data: pd.DataFrame,
    experiments: list[FeatureEngineeringExperiment],
    target: str = TARGET,
    cv: KFold | None = None,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = 2,
) -> pd.DataFrame:
    cv = cv or make_cv()

    X = data.drop(columns=[target])
    y = data[target]

    results = []

    for experiment in experiments:
        scores = cross_validate(
            experiment.estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )

        rmse_scores = -scores["test_score"]

        results.append(
            {
                "experiment": experiment.name,
                "description": experiment.description,
                "mean_rmse": rmse_scores.mean(),
                "std_rmse": rmse_scores.std(),
            }
        )

    results = pd.DataFrame(results)
    results = results.sort_values("mean_rmse", ascending=True)

    baseline_rmse = results.loc[
        results["experiment"] == "baseline_hgbr",
        "mean_rmse",
    ].iloc[0]

    results["delta_vs_baseline"] = results["mean_rmse"] - baseline_rmse

    return results


def plot_feature_engineering_results(
    results: pd.DataFrame,
    figsize: tuple[int, int] = (10, 6),
):
    import matplotlib.pyplot as plt

    plot_data = results.sort_values("mean_rmse", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        plot_data["experiment"],
        plot_data["mean_rmse"],
    )

    ax.invert_yaxis()
    ax.set_title("Feature engineering experiments")
    ax.set_xlabel("CV RMSE")
    ax.set_ylabel("Experiment")

    for index, value in enumerate(plot_data["mean_rmse"]):
        ax.text(
            value,
            index,
            f" {value:.2f}",
            va="center",
        )

    fig.tight_layout()

    return fig, ax