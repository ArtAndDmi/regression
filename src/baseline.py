import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=['number']
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                'cat',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                'num',
                'passthrough',
                numeric_features
            )
        ]
    )


def get_baseline_models(n_jobs: int = 2) -> dict:
    return {
        'Dummy': DummyRegressor(strategy='mean'),

        'Linear Regression': LinearRegression(),

        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs
        ),

        'Hist Gradient Boosting': HistGradientBoostingRegressor(
            random_state=42
        )
    }


def build_pipeline(
    model,
    X: pd.DataFrame
) -> Pipeline:
    return Pipeline(
        steps=[
            ('preprocessor', build_preprocessor(X)),
            ('model', model)
        ]
    )


def run_baseline_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict | None = None,
    cv_splits: int = 5,
    n_jobs: int = 2
) -> pd.DataFrame:
    if models is None:
        models = get_baseline_models(n_jobs=n_jobs)

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    results = []

    for model_name, model in models.items():
        print(f'Evaluating: {model_name}')

        pipeline = build_pipeline(
            model=model,
            X=X_train
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=n_jobs
        )

        rmse_scores = -scores

        results.append({
            'model': model_name,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std()
        })

    return (
        pd.DataFrame(results)
        .sort_values('cv_rmse_mean')
        .reset_index(drop=True)
    )


def plot_baseline_results(
    baseline_results: pd.DataFrame
) -> None:
    plt.figure(figsize=(10, 5))

    plt.bar(
        baseline_results['model'],
        baseline_results['cv_rmse_mean'],
        yerr=baseline_results['cv_rmse_std'],
        capsize=5
    )

    plt.title('Baseline models comparison')
    plt.ylabel('CV RMSE')
    plt.xticks(rotation=30)
    plt.show()


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model
) -> Pipeline:
    pipeline = build_pipeline(
        model=model,
        X=X_train
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def plot_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    n_repeats: int = 5,
    n_jobs: int = 2
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring='neg_root_mean_squared_error',
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=n_jobs
    )

    importance_df = (
        pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        .sort_values('importance_mean', ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df['feature'][::-1],
        importance_df['importance_mean'][::-1],
        xerr=importance_df['importance_std'][::-1]
    )

    plt.title('Permutation Importance')
    plt.xlabel('Importance')
    plt.show()

    return importance_df


def plot_actual_vs_predicted(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_size: float = 0.2
) -> None:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=42
    )

    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_val)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_val,
        y_pred,
        alpha=0.3
    )

    min_value = min(y_val.min(), y_pred.min())
    max_value = max(y_val.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()


def evaluate_on_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_size: float = 0.2
) -> dict:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=42
    )

    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_val)

    return {
        'RMSE': root_mean_squared_error(y_val, y_pred),
        'MAE': mean_absolute_error(y_val, y_pred),
        'R2': r2_score(y_val, y_pred)
    }