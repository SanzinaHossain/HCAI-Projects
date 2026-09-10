from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class TrainingResult:
    problem_type: str
    model_label: str
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: dict[str, float]
    interpretation: str
    confusion_matrix: list[list[int]] | None = None
    class_names: list[str] | None = None
    classification_report: list[dict[str, Any]] | None = None
    predictions_preview: list[dict[str, Any]] | None = None


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbours",
    "svm": "Support Vector Machine",
}


def detect_problem_type(y: pd.Series) -> str:
    """Use a simple, explainable rule to identify classification vs regression."""
    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        return "classification"

    unique_count = y.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y), 1)

    if unique_count <= 20 or unique_ratio < 0.08:
        return "classification"
    return "regression"


def _build_model(model_name: str, problem_type: str, fit_intercept: bool):
    if problem_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                fit_intercept=fit_intercept,
                random_state=42,
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
            ),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(probability=True, random_state=42),
        }
    else:
        models = {
            "logistic_regression": LinearRegression(
                fit_intercept=fit_intercept
            ),
            "decision_tree": DecisionTreeRegressor(
                max_depth=5,
                random_state=42,
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
            ),
            "knn": KNeighborsRegressor(n_neighbors=5),
            "svm": SVR(),
        }

    if model_name not in models:
        raise ValueError("The selected model is not supported.")
    return models[model_name]


def _interpret_score(problem_type: str, score: float) -> str:
    if problem_type == "classification":
        if score >= 0.90:
            return "Excellent: the model predicts most test examples correctly."
        if score >= 0.80:
            return "Good: the model performs well, although some mistakes remain."
        if score >= 0.65:
            return "Fair: the model finds useful patterns, but it should be improved."
        return "Weak: the model is struggling to generalize to unseen data."

    if score >= 0.85:
        return "Excellent: the model explains most of the variation in the target."
    if score >= 0.65:
        return "Good: the model captures a useful part of the pattern."
    if score >= 0.35:
        return "Fair: the model captures some signal, but predictions may be uncertain."
    return "Weak: the model does not yet explain the target reliably."


def train_model(
    dataframe: pd.DataFrame,
    target_column: str,
    model_name: str,
    test_size_percent: int,
    normalize: bool,
    fit_intercept: bool,
) -> TrainingResult:
    if target_column not in dataframe.columns:
        raise ValueError("Please select a valid target column.")

    clean_df = dataframe.copy()
    clean_df = clean_df.dropna(subset=[target_column])

    if clean_df.empty:
        raise ValueError("The selected target column contains no usable values.")

    X = clean_df.drop(columns=[target_column])
    y = clean_df[target_column]

    if X.shape[1] == 0:
        raise ValueError("The dataset needs at least one input column.")

    problem_type = detect_problem_type(y)

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if normalize:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = _build_model(model_name, problem_type, fit_intercept)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    stratify = None
    if problem_type == "classification":
        value_counts = y.value_counts()
        if len(value_counts) > 1 and value_counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_percent / 100,
        random_state=42,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    preview = [
        {
            "actual": str(actual),
            "predicted": str(predicted),
        }
        for actual, predicted in list(zip(y_test.tolist(), predictions.tolist()))[:8]
    ]

    if problem_type == "classification":
        accuracy = accuracy_score(y_test, predictions)
        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
        original_labels = sorted(
            set(y_test.tolist()) | set(predictions.tolist()),
            key=lambda value: str(value),
        )
        labels = [str(value) for value in original_labels]

        report_dict = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        report_rows = []
        for label, values in report_dict.items():
            if isinstance(values, dict):
                report_rows.append(
                    {
                        "label": str(label),
                        "precision": round(float(values["precision"]) * 100, 1),
                        "recall": round(float(values["recall"]) * 100, 1),
                        "f1": round(float(values["f1-score"]) * 100, 1),
                        "support": int(values["support"]),
                    }
                )

        return TrainingResult(
            problem_type=problem_type,
            model_label=MODEL_LABELS[model_name],
            primary_metric_name="Accuracy",
            primary_metric_value=round(accuracy * 100, 1),
            secondary_metrics={
                "Weighted F1 score": round(weighted_f1 * 100, 1),
                "Training rows": float(len(X_train)),
                "Testing rows": float(len(X_test)),
            },
            interpretation=_interpret_score(problem_type, accuracy),
            confusion_matrix=confusion_matrix(
                y_test,
                predictions,
                labels=original_labels,
            ).tolist() if len(original_labels) else None,
            class_names=labels,
            classification_report=report_rows,
            predictions_preview=preview,
        )

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return TrainingResult(
        problem_type=problem_type,
        model_label=MODEL_LABELS[model_name],
        primary_metric_name="R² score",
        primary_metric_value=round(r2 * 100, 1),
        secondary_metrics={
            "Mean absolute error": round(float(mae), 3),
            "Root mean squared error": round(float(rmse), 3),
            "Training rows": float(len(X_train)),
            "Testing rows": float(len(X_test)),
        },
        interpretation=_interpret_score(problem_type, r2),
        predictions_preview=preview,
    )
