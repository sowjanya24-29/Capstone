"""Train and compare Titanic models on the CSV produced by 01_eda.py.

Does not call sns.load_dataset. Preprocessing is fit on the training split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "titanic.csv"
FIG = ROOT / "figures"
REPORT_PATH = ROOT / "modeling_report.json"
PIPELINE_PATH = ROOT / "survival_pipeline.joblib"

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC = ["age", "fare", "sibsp", "parch", "pclass"]
CATEGORICAL = ["sex", "embarked"]
TARGET = "survived"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc": round(float(auc(fpr, tpr)), 4),
        "fpr": [round(float(v), 4) for v in fpr],
        "tpr": [round(float(v), 4) for v in tpr],
    }


def fit_classifier(name: str, estimator, x_train, y_train, x_test, y_test) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    metrics["model"] = name
    return pipeline, metrics


def plot_roc(curves: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for item in curves:
        ax.plot(item["fpr"], item["tpr"], label=f"{item['model']} (AUC={item['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png", dpi=120)
    plt.close(fig)


def plot_decision_tree(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["not_survived", "survived"],
        filled=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision tree (top 3 levels of the fitted tree)")
    fig.tight_layout()
    fig.savefig(FIG / "decision_tree.png", dpi=120)
    plt.close(fig)


def imbalance_comparison(x_train, y_train, x_test, y_test) -> list[dict]:
    """Retrain logistic regression three ways. SMOTE sees only the training fold."""
    results = []
    baseline, metrics = fit_classifier(
        "logreg_baseline",
        LogisticRegression(max_iter=500),
        x_train,
        y_train,
        x_test,
        y_test,
    )
    results.append({k: metrics[k] for k in ("model", "precision", "recall", "f1")})

    _, metrics = fit_classifier(
        "logreg_class_weight",
        LogisticRegression(max_iter=500, class_weight="balanced"),
        x_train,
        y_train,
        x_test,
        y_test,
    )
    results.append({k: metrics[k] for k in ("model", "precision", "recall", "f1")})

    preprocessor = build_preprocessor()
    x_train_ready = preprocessor.fit_transform(x_train)
    x_test_ready = preprocessor.transform(x_test)
    sampler = SMOTE(random_state=42)
    x_resampled, y_resampled = sampler.fit_resample(x_train_ready, y_train)
    smote_model = LogisticRegression(max_iter=500)
    smote_model.fit(x_resampled, y_resampled)
    pred = smote_model.predict(x_test_ready)
    prob = smote_model.predict_proba(x_test_ready)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    results.append(
        {
            "model": "logreg_smote_train_only",
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "train_rows_before": int(len(y_train)),
            "train_rows_after_smote": int(len(y_resampled)),
        }
    )
    del baseline
    return results


def tune_forest(x_train, y_train) -> dict:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(oob_score=True, random_state=42),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 8, None],
            "model__max_features": ["sqrt", "log2"],
        },
        cv=3,
        scoring="f1",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_.named_steps["model"]
    return {
        "best_params": search.best_params_,
        "best_cv_f1": round(float(search.best_score_), 4),
        "oob_score": round(float(best.oob_score_), 4),
    }


def regression_task(frame: pd.DataFrame) -> dict:
    predictors = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
    work = frame[predictors + ["fare"]].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        work[predictors], work["fare"], test_size=0.2, random_state=42
    )
    numeric = ["age", "sibsp", "parch", "pclass", "survived"]
    categorical = ["sex", "embarked"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    residual = y_test.to_numpy() - pred
    r2 = float(r2_score(y_test, pred))
    n = len(y_test)
    p = model.named_steps["preprocess"].get_feature_names_out().shape[0]
    adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    # A rising spread of |residual| against the fitted value is heteroscedasticity.
    spread_corr = float(np.corrcoef(np.abs(residual), pred)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, residual, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Fitted fare")
    ax.set_ylabel("Residual")
    ax.set_title("Fare regression residuals")
    fig.tight_layout()
    fig.savefig(FIG / "fare_residuals.png", dpi=120)
    plt.close(fig)

    return {
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "r2": round(r2, 4),
        "adjusted_r2": round(float(adjusted), 4),
        "abs_residual_vs_fitted_corr": round(spread_corr, 4),
        "heteroscedasticity": "present" if spread_corr > 0.2 else "not clearly present",
    }


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit("titanic.csv is missing. Run 01_eda.py first.")
    FIG.mkdir(exist_ok=True)
    frame = pd.read_csv(CSV_PATH)
    x = frame[FEATURES]
    y = frame[TARGET]
    balance = {
        "not_survived": int((y == 0).sum()),
        "survived": int((y == 1).sum()),
        "survived_share": round(float(y.mean()), 4),
    }
    # Stratify because survival is the minority class (~38%). An unstratified
    # split can move that rate between folds and make accuracy look unstable.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    specs = [
        ("logistic_regression", LogisticRegression(max_iter=500)),
        ("decision_tree", DecisionTreeClassifier(random_state=42)),
        ("random_forest", RandomForestClassifier(random_state=42)),
    ]
    fitted = {}
    metrics = []
    for name, estimator in specs:
        pipeline, scores = fit_classifier(name, estimator, x_train, y_train, x_test, y_test)
        fitted[name] = pipeline
        metrics.append(scores)
        print(name, {k: scores[k] for k in ("accuracy", "precision", "recall", "f1", "auc")})

    plot_roc(metrics)
    plot_decision_tree(fitted["decision_tree"])
    imbalance = imbalance_comparison(x_train, y_train, x_test, y_test)
    tuning = tune_forest(x_train, y_train)
    regression = regression_task(frame)

    best_name = max(metrics, key=lambda item: (item["f1"], item["auc"]))["model"]
    best_pipeline = fitted[best_name]
    joblib.dump(best_pipeline, PIPELINE_PATH)
    reloaded = joblib.load(PIPELINE_PATH)
    sample = x_test.head(1)
    original = float(best_pipeline.predict(sample)[0])
    reloaded_pred = float(reloaded.predict(sample)[0])

    report = {
        "class_balance": balance,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_survived_share": round(float(y_train.mean()), 4),
        "test_survived_share": round(float(y_test.mean()), 4),
        "preprocessing": {
            "numeric": "median impute + standard scale, fit on train only",
            "categorical": "most-frequent impute + one-hot, fit on train only",
            "why_it_differs_from_eda": (
                "EDA drops rare missing rows and the deck column for the story. "
                "The saved model must score raw rows, so imputation stays inside "
                "the pipeline instead of deleting rows before the split."
            ),
        },
        "classifiers": [
            {k: item[k] for k in ("model", "confusion_matrix", "accuracy", "precision", "recall", "f1", "auc")}
            for item in metrics
        ],
        "imbalance": imbalance,
        "random_forest_search": tuning,
        "regression": regression,
        "deploy_candidate": best_name,
        "reload_check": {
            "original_prediction": original,
            "reloaded_prediction": reloaded_pred,
            "match": original == reloaded_pred,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.name}; reload match: {report['reload_check']['match']}")


if __name__ == "__main__":
    main()
