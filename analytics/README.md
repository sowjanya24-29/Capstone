# Analytics pipeline

One Titanic pass: profile the raw file, clean it for the story, then model survival and fare from that same file. `02_modeling.py` never calls `sns.load_dataset`.

## Setup

```powershell
cd analytics
python -m pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py
```

Notebook equivalents of the same pipeline (run from the `analytics` folder, EDA first):

- `01_eda.ipynb`
- `02_modeling.ipynb`

`01_eda.py` / `01_eda.ipynb` load Seaborn's Titanic dataset once and write `titanic.csv`. Later runs, and grading without network, read that file with `pd.read_csv("titanic.csv")`. Charts land in `figures/`. Numbers below come from `eda_summary.json` and `modeling_report.json`.

## Missing values

Measured on the raw 891 × 15 frame before any cleaning:

| Column | Missing | Rule | Action |
| --- | --- | --- | --- |
| age | 19.8653% | 5%–30%: impute | median 28.0 |
| embarked | 0.2245% | under 5%: drop those rows | dropped |
| embark_town | 0.2245% | under 5%: drop those rows | dropped |
| deck | 77.2166% | too incomplete to impute | column dropped |

`deck` is a cabin letter that is empty for about 77% of passengers. Filling it would invent the majority of the column, so the column is removed. Dropping the two nearly empty embarkation rows leaves 889 passengers. No other column has missing values.

## Outliers and fare shape

IQR fences are Q1 − 1.5×IQR and Q3 + 1.5×IQR, computed on observed values (age uses the 714 non-missing ages, before the median fill).

- Age: Q1 20.125, Q3 38.0, fence above 64.8125. **11 outliers.**
- Fare: Q1 7.9104, Q3 31.0, fence above 65.6344. **116 outliers.**

Fare mean 32.20, median 14.45, mode 8.05. Mean > median > mode, so the fare distribution is **right-skewed**. A small number of expensive tickets pull the mean above the typical fare.

## Survival rates

Computed with boolean masks on the cleaned frame.

- By sex: female 0.7404, male 0.1889.
- By class: 1st 0.6262, 2nd 0.4728, 3rd 0.2424.
- By sex and class: first-class women 0.9674, second-class women 0.9211, third-class women 0.5000, first-class men 0.3689, second-class men 0.1574, third-class men 0.1354.

## Correlation

The heatmap uses only `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. `adult_male` and `alone` are left out because they are flags derived from sex/age and from sibsp+parch.

The two strongest off-diagonal pairs, ranked by absolute correlation:

1. **pclass and fare, −0.5482.** A higher class number is a lower ticket class, and those passengers paid less. Fare and class describe the same wealth split more than they describe two independent causes.
2. **sibsp and parch, 0.4145.** Passengers traveling with siblings or a spouse also tend to travel with parents or children. These are family parties, not separate effects.

## Charts

`figures/univariate_age_fare.png` holds the age and fare histograms and box plots. `figures/correlation_heatmap.png` is the 6×6 matrix. `figures/standardization_age_fare.png` is the z-score check. The four story charts:

**Survival rate by sex** (`story_survival_by_sex.png`). Women survived at 74.0% and men at 18.9%. That gap is larger than any age effect in the correlation matrix (age vs survived is only −0.07). Sex is the first split in the survival story, consistent with women being given boats before men.

**Survival rate by class and sex** (`story_survival_by_class_sex.png`). First- and second-class women survived at 96.7% and 92.1%, while third-class women survived at 50%. Men were lower in every class, and a first-class man (36.9%) still survived less often than a third-class woman. Class shifts the level. Sex changes the outcome more.

**Fare by class and survival** (`story_fare_by_class.png`). Ticket prices fall from first class to third, and survivor boxes sit a little higher inside a class, but the boxes overlap and 116 fares are IQR outliers. Fare is mostly a noisy restatement of class. The right skew (mean well above the mode) is why the upper whiskers are so long.

**Age versus fare, colored by survival** (`story_age_fare_survival.png`). Survivors appear at every age. The expensive part of the plot, which is mostly first class, has a denser mix of survivors, including both children and adults. There is no single age cutoff. The 11 age outliers are the passengers older than 64.8 years, a thin slice next to the fare outliers.

## Standardization check

This is an EDA check only. It is fit on the full cleaned frame and is not the scaler inside the model.

z = (x − mean) / std via `StandardScaler` on age and fare. Before: age mean 29.3152 (std 12.9849), fare mean 32.0967 (std 49.6975). After: both means are 0.0 and both standard deviations are 1.0006. The slight difference from 1 is sample standard deviation (divide by n−1) on values that were scaled with the population standard deviation (divide by n).

## Modeling

Class balance on the raw file: 549 did not survive, 342 did (38.38% survived). The split is stratified 80/20 (`random_state=42`) because an unstratified cut can move that minority share between train and test and make accuracy hard to compare. The split kept the rate: train 38.34% survived (712 rows), test 38.55% (179 rows).

Preprocessing is a `ColumnTransformer` inside a `Pipeline`, fit on the training split only. Numeric columns (`age`, `fare`, `sibsp`, `parch`, `pclass`) are median-imputed and scaled. `sex` and `embarked` are most-frequent-imputed and one-hot encoded. This is deliberately not the EDA rule. EDA drops rare missing rows for the charts. The saved model has to score raw rows, including a missing age or embarkation port, so imputation stays inside the pipeline and no preprocessor is refit on the test set or on the full frame.

`alive` is not a feature. It repeats `survived`. `deck` is not a feature. It is mostly missing.

### Classifier comparison

Positive class is survived = 1.

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision tree | 0.8156 | 0.7727 | 0.7391 | 0.7556 | 0.7967 |
| Random forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8271 |

Confusion matrices (rows true, columns predicted, order not-survived then survived):

- Logistic regression: [[98, 12], [23, 46]]
- Decision tree: [[95, 15], [18, 51]]
- Random forest: [[98, 12], [21, 48]]

ROC curves are in `figures/roc_curves.png`. The fitted decision tree is drawn with feature names and class names (`not_survived`, `survived`) in `figures/decision_tree.png`. The figure shows the top three levels so the labels stay readable; the scored model is the full fitted tree.

### Imbalance

The same logistic regression was retrained three ways. SMOTE was fit on the transformed training fold only (712 rows became 878). The test fold was not resampled.

| Variant | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Baseline | 0.7931 | 0.6667 | 0.7244 |
| class_weight='balanced' | 0.7297 | 0.7826 | 0.7552 |
| SMOTE, train only | 0.7397 | 0.7826 | 0.7606 |

SMOTE is the better of the three for this model. It raises recall from 0.6667 to 0.7826 and F1 from 0.7244 to 0.7606. `class_weight='balanced'` matches that recall but lands a slightly lower F1 (0.7552) because precision falls further (0.7297 vs 0.7397). The baseline is the most precise and the worst at finding survivors. For a survival decision, missing a survivor is the costlier error, so the F1 gain from SMOTE is worth the precision drop.

### Random forest search

`GridSearchCV` (3-fold, scoring F1) on `RandomForestClassifier(oob_score=True, random_state=42)` over `n_estimators`, `max_depth`, and `max_features`.

- Best parameters: `max_depth=8`, `max_features='sqrt'`, `n_estimators=200`
- Best cross-validated F1: 0.7479
- Out-of-bag score: 0.8244

### Fare regression

Multivariate linear regression of `fare` on `pclass`, `sex`, `age`, `sibsp`, `parch`, `embarked`, and `survived`. Preprocessing is again fit on the training split only.

| MAE | RMSE | R² | Adjusted R² |
| --- | --- | --- | --- |
| 20.8977 | 30.5328 | 0.3975 | 0.3617 |

These are not on the same scale as the classifier metrics and are not compared with them. The residual plot (`figures/fare_residuals.png`) fans out as the fitted fare grows. Correlation between absolute residual and fitted value is 0.5456, so the errors are **heteroscedastic**. That matches the right-skewed fare distribution: large tickets are harder to predict in pounds than cheap ones.

### Recommendation

I would deploy the **decision tree**. On the shared test split it ties the random forest for the best accuracy (0.8156) and has the best F1 (0.7556) and the best recall (0.7391) of the three untuned classifiers. Logistic regression ranks passengers better (AUC 0.8437 vs 0.7967) but misses more survivors (recall 0.6667). If the decision is "who gets a boat," the extra true positives matter more than the ranking score. The tree is the saved artifact: `survival_pipeline.joblib` is the preprocessing pipeline plus this tree, not the bare estimator. Reloading it with `joblib.load` reproduces the same prediction on a raw test row.

The imbalance check is a reason to revisit that choice later. SMOTE logistic regression reaches F1 0.7606, just above the tree, but that was a side comparison and is not the saved pipeline.
