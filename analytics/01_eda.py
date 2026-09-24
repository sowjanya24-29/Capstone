"""Profile, clean, and visualize the Titanic dataset.

The raw frame is loaded once (Seaborn, then cached) and written to titanic.csv.
Modeling must read that file and must not call sns.load_dataset again.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "titanic.csv"
FIG = ROOT / "figures"
SUMMARY_PATH = ROOT / "eda_summary.json"

STORY_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]


def load_raw() -> pd.DataFrame:
    if CSV_PATH.exists():
        print(f"Reusing committed offline file {CSV_PATH.name}")
        return pd.read_csv(CSV_PATH)
    frame = sns.load_dataset("titanic")
    frame.to_csv(CSV_PATH, index=False)
    print(f"Loaded Titanic once via sns.load_dataset and saved {CSV_PATH.name}")
    return frame


def missing_report(frame: pd.DataFrame) -> dict[str, float]:
    pct = (frame.isna().mean() * 100).round(4)
    return {col: float(pct[col]) for col in pct.index if pct[col] > 0}


def apply_missing_strategy(frame: pd.DataFrame, missing: dict[str, float]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the assignment thresholds and record the decision for each column.

    Under 5% missing: drop those rows.
    5% to 30% missing: median-impute numeric columns.
    Above 30%: drop the column. Imputing a column that is mostly empty would
    invent values for the majority of passengers (deck is a cabin letter, not a
    quantity we can responsibly fill).
    """
    decisions = []
    cleaned = frame.copy()
    drop_rows_for: list[str] = []
    impute_cols: list[str] = []
    drop_cols: list[str] = []

    for column, pct in missing.items():
        if pct < 5:
            action = "drop_rows"
            drop_rows_for.append(column)
        elif pct <= 30:
            action = "median_impute"
            impute_cols.append(column)
        else:
            action = "drop_column"
            drop_cols.append(column)
        decisions.append({"column": column, "missing_pct": pct, "action": action})

    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)
    if drop_rows_for:
        cleaned = cleaned.dropna(subset=drop_rows_for)
    for column in impute_cols:
        if column not in cleaned.columns:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median = float(cleaned[column].median())
            cleaned[column] = cleaned[column].fillna(median)
            for item in decisions:
                if item["column"] == column:
                    item["impute_value"] = median
        else:
            mode = cleaned[column].mode(dropna=True).iloc[0]
            cleaned[column] = cleaned[column].fillna(mode)
            for item in decisions:
                if item["column"] == column:
                    item["impute_value"] = str(mode)
    return cleaned.reset_index(drop=True), decisions


def iqr_outlier_count(series: pd.Series) -> dict:
    values = series.dropna()
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = int(((values < low) | (values > high)).sum())
    return {"q1": q1, "q3": q3, "iqr": iqr, "low": low, "high": high, "outliers": count}


def fare_shape(series: pd.Series) -> dict:
    mean = float(series.mean())
    median = float(series.median())
    mode = float(series.mode().iloc[0])
    if mean > median > mode or mean > median:
        shape = "right-skewed"
    elif mean < median < mode or mean < median:
        shape = "left-skewed"
    else:
        shape = "symmetric"
    return {"mean": mean, "median": median, "mode": mode, "shape": shape}


def survival_tables(frame: pd.DataFrame) -> dict:
    def rate(mask: pd.Series) -> float:
        subset = frame.loc[mask, "survived"]
        return round(float(subset.mean()), 4) if len(subset) else None

    by_sex = {sex: rate(frame["sex"] == sex) for sex in sorted(frame["sex"].unique())}
    by_class = {str(pclass): rate(frame["pclass"] == pclass) for pclass in sorted(frame["pclass"].unique())}
    by_both = {}
    for sex in sorted(frame["sex"].unique()):
        for pclass in sorted(frame["pclass"].unique()):
            key = f"{sex}_pclass_{pclass}"
            by_both[key] = rate((frame["sex"] == sex) & (frame["pclass"] == pclass))
    return {"by_sex": by_sex, "by_pclass": by_class, "by_sex_and_pclass": by_both}


def strongest_correlations(corr: pd.DataFrame) -> list[dict]:
    pairs = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(corr.loc[left, right])
            pairs.append({"a": left, "b": right, "correlation": round(value, 4), "abs": abs(value)})
    pairs.sort(key=lambda item: item["abs"], reverse=True)
    return pairs[:2]


def save_univariate(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.histplot(frame["age"], bins=20, ax=axes[0, 0])
    axes[0, 0].set_title("Age histogram")
    sns.boxplot(x=frame["age"], ax=axes[0, 1])
    axes[0, 1].set_title("Age box plot")
    sns.histplot(frame["fare"], bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Fare histogram")
    sns.boxplot(x=frame["fare"], ax=axes[1, 1])
    axes[1, 1].set_title("Fare box plot")
    fig.tight_layout()
    fig.savefig(FIG / "univariate_age_fare.png", dpi=120)
    plt.close(fig)


def save_story_charts(frame: pd.DataFrame, corr: pd.DataFrame) -> None:
    sex_rates = frame.groupby("sex")["survived"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sex_rates, x="sex", y="survived", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by sex")
    fig.tight_layout()
    fig.savefig(FIG / "story_survival_by_sex.png", dpi=120)
    plt.close(fig)

    class_sex = frame.groupby(["pclass", "sex"])["survived"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=class_sex, x="pclass", y="survived", hue="sex", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by class and sex")
    fig.tight_layout()
    fig.savefig(FIG / "story_survival_by_class_sex.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=frame, x="pclass", y="fare", hue="survived", ax=ax)
    ax.set_title("Fare by class and survival")
    fig.tight_layout()
    fig.savefig(FIG / "story_fare_by_class.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=frame, x="age", y="fare", hue="survived", alpha=0.7, ax=ax)
    ax.set_title("Age vs fare, colored by survival")
    fig.tight_layout()
    fig.savefig(FIG / "story_age_fare_survival.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation of survived, pclass, age, sibsp, parch, fare")
    fig.tight_layout()
    fig.savefig(FIG / "correlation_heatmap.png", dpi=120)
    plt.close(fig)


def standardization_check(frame: pd.DataFrame) -> dict:
    before = frame[["age", "fare"]].agg(["mean", "std"]).round(4)
    scaled = StandardScaler().fit_transform(frame[["age", "fare"]])
    scaled_frame = pd.DataFrame(scaled, columns=["age", "fare"])
    after = scaled_frame.agg(["mean", "std"]).round(4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.kdeplot(frame["fare"], ax=axes[0], label="original")
    sns.kdeplot(scaled_frame["fare"], ax=axes[0], label="z-score")
    axes[0].legend()
    axes[0].set_title("Fare before and after z-score")
    sns.kdeplot(frame["age"], ax=axes[1], label="original")
    sns.kdeplot(scaled_frame["age"], ax=axes[1], label="z-score")
    axes[1].legend()
    axes[1].set_title("Age before and after z-score")
    fig.tight_layout()
    fig.savefig(FIG / "standardization_age_fare.png", dpi=120)
    plt.close(fig)
    return {"before": before.to_dict(), "after": after.to_dict()}


def main() -> None:
    FIG.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    raw = load_raw()
    profile = {
        "shape": list(raw.shape),
        "dtypes": {col: str(dtype) for col, dtype in raw.dtypes.items()},
        "describe": json.loads(raw.describe(include="all").to_json()),
    }
    print("shape", raw.shape)
    raw.info()
    print(raw.describe(include="all"))

    missing = missing_report(raw)
    print("missing_pct", missing)
    age_outliers = iqr_outlier_count(raw["age"])
    fare_outliers = iqr_outlier_count(raw["fare"])
    fare_stats = fare_shape(raw["fare"].dropna())

    cleaned, decisions = apply_missing_strategy(raw, missing)
    rates = survival_tables(cleaned)
    corr = cleaned[STORY_COLUMNS].corr(numeric_only=True).round(4)
    top_corr = strongest_correlations(corr)
    save_univariate(cleaned)
    save_story_charts(cleaned, corr)
    scaling = standardization_check(cleaned)

    summary = {
        "profile_shape": profile["shape"],
        "missing_pct": missing,
        "missing_decisions": decisions,
        "rows_after_cleaning": int(len(cleaned)),
        "age_outliers_iqr": age_outliers,
        "fare_outliers_iqr": fare_outliers,
        "fare_shape": fare_stats,
        "survival": rates,
        "correlation": corr.round(4).to_dict(),
        "strongest_correlations": top_corr,
        "standardization": scaling,
        "class_balance": {
            "survived_rate": round(float(raw["survived"].mean()), 4),
            "counts": {str(k): int(v) for k, v in raw["survived"].value_counts().to_dict().items()},
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
