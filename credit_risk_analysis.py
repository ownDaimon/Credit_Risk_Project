import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
df = pd.read_csv(r"C:\Users\Gem\Downloads\Loan_Data1.csv")
#df = pd.read_csv('Loan_Data1.csv')
df_original = df.copy()
# import os
# print(os.getcwd())


#MODERN ML PIPELINE



# =====================================
# PART 1 : EXPLORATORY DATA ANALYSIS (EDA)
# =====================================

# FYI our data is clean but I've still included EDA

print("="*50)
print("DATASET OVERVIEW")
print("="*50)

print(f"Number of observations : {df.shape[0]}")
print(f"Number of features     : {df.shape[1]}")

print("\nData Types")
print(df.dtypes)

print("\nFirst Five Rows")
print(df.head())

print("\nSummary Statistics")
print(df.describe())

# =====================================
# Missing Value Analysis
# =====================================

print("\n" + "="*50)
print("MISSING VALUE ANALYSIS")
print("="*50)

missing_values = df.isnull().sum()

missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage": (missing_values / len(df)) * 100
})

print(missing_summary)

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Value Heatmap")
plt.show()


# =====================================
# Class imbalance check
# =====================================
print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)

class_counts = df['default'].value_counts()

print(class_counts)

plt.figure(figsize=(6,4))
sns.countplot(x='default', data=df)

plt.title("Distribution of Default Classes")
plt.xlabel("Default")
plt.ylabel("Number of Borrowers")
plt.show()

print("\nClass Percentage")

print((class_counts/len(df))*100)



# =====================================
# Outlier check
# =====================================
print("\n" + "="*50)
print("OUTLIER DETECTION plot")
print("="*50)

numeric_columns = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(16,10))

for i, column in enumerate(numeric_columns, 1):
    plt.subplot((len(numeric_columns)+2)//3, 3, i)
    sns.boxplot(y=df[column])
    plt.title(column)

plt.tight_layout()
plt.show()

# =====================================================
# Outlier Detection (IQR Method)
# =====================================================
#Included this part for the sake of completance as well. 
#These are not really outliers as they have reasonable explanations

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("\n========== OUTLIER ANALYSIS ==========\n")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\n{column}")
    print("-" * 50)
    print(f"Number of Outliers : {len(outliers)}")

    if len(outliers) > 0:
        print(outliers[[column]].head(10))
        
      

# =====================================================
# Distribution of KKB Scores
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    df['kkb_score'],
    bins=25,
    edgecolor='black'
)

plt.title("Distribution of KKB Scores")
plt.xlabel("KKB Score")
plt.ylabel("Frequency")

plt.show()

print("\n========== KKB SCORE SUMMARY ==========\n")
print(df['kkb_score'].describe())

print("\nMedian KKB Score:", df['kkb_score'].median())
print("Mode KKB Score:", df['kkb_score'].mode()[0])


features = [
    "kkb_score",
    "income",
    "loan_amount_o",
    "total_debt_o"
]

# Features vs Default 
for feature in features:

    plt.figure(figsize=(7,4))

    sns.boxplot(
        x="default",
        y=feature,
        data=df
    )

    plt.title(f"{feature} by Default Status")
    plt.show()

# =====================================================
# FEATURE ENGINEERING
# =====================================================

df["dti"] = df["total_debt_o"] / df["income"]
#print("DEBT TO INCOME RATIO")
#print(df["dti"].head())

df["loan_income_ratio"] = df["loan_amount_o"] / df["income"]
#print("LOAN TO INCOME RATIO")
#print(df["loan_income_ratio"].head())


df["debt_per_credit_line"] = (df["total_debt_o"] /df["credit_lines_outstanding"])

df["loan_share"] = (df["loan_amount_o"] /df["total_debt_o"])

df["income_per_credit_line"] = (df["income"] / df["credit_lines_outstanding"])

df["employment_group"] = pd.cut(
    df["years_employed"],
    bins=[0,2,5,10,100],
    labels=[
        "New",
        "Experienced",
        "Stable",
        "Veteran"
    ]
)

#Checking for infinity because of received error
for col in df.select_dtypes(include=np.number):

    if np.isinf(df[col]).any():
        print(col)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =====================================================
# Correlation Matrix
# =====================================================

plt.figure(figsize=(10,8))

correlation_matrix = df.drop(columns=['customer_id']).corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("\n========== CORRELATION MATRIX ==========\n")
print(correlation_matrix.round(2))       
        
#These correlations are expected  


# =====================================
# ONE HOT ENCODER FOR CAT VARIABLES
# =====================================
df = pd.get_dummies(df,columns=["employment_group"],drop_first=True)


# =====================================
# PART 2 : CREDIT RISK MODEL DEVELOPMENT
# =====================================

print("\n" + "="*60)
print("PART 2 : CREDIT RISK MODEL DEVELOPMENT")
print("="*60)

X = df.drop(['default', 'customer_id'], axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#Cross validation
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



# Model-1 BASE MODEL: Logistic Regression for calculating probability of default

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


logistic_cv = LogisticRegressionCV(

    Cs=np.logspace(-4, 4, 40),

    cv=inner_cv,

    penalty="l2",

    solver="lbfgs",

    scoring="roc_auc",

    max_iter=5000,

    random_state=42,

    n_jobs=-1
)


#pipeline = Pipeline([("scaler", StandardScaler()),("logistic", logistic_cv)])
from sklearn.impute import KNNImputer

pipeline = Pipeline([
    ("imputer", KNNImputer(
        n_neighbors=5,
        weights="distance"
    )),
    ("scaler", StandardScaler()),
    ("logistic", logistic_cv)
])

#Distance because: Nearby observations influence the imputed value more. In credit data, borrowers that are very similar should contribute more than distant ones. It's a common improvement over the default.

outer_cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)



nested_results = cross_validate(

    estimator=pipeline,

    X=X_train,

    y=y_train,

    cv=outer_cv,

    scoring="roc_auc",

    return_train_score=True,

    n_jobs=-1

)


print("=" * 60)
print("NESTED CROSS VALIDATION RESULTS")
print("=" * 60)

print("Fold AUC Scores")

for i, score in enumerate(nested_results["test_score"], 1):
    print(f"Fold {i}: {score:.4f}")

print()

print(f"Mean CV AUC : {nested_results['test_score'].mean():.4f}")
print(f"Std CV AUC  : {nested_results['test_score'].std():.4f}")


# Final Model
pipeline.fit(X_train, y_train)

# Hyperparameter optimization part for logistic regression
best_C = pipeline.named_steps["logistic"].C_[0]

print()
print(f"Optimal C selected by CV: {best_C:.6f}")

#Evaluate on the Test Set
y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"Test ROC AUC : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Brier Score  : {brier_score_loss(y_test, y_prob):.4f}")


from sklearn.metrics import log_loss

# McFadden's Pseudo R²
ll_model = -log_loss(y_test, y_prob, normalize=False)

p_null = np.repeat(y_train.mean(), len(y_test))
ll_null = -log_loss(y_test, p_null, normalize=False)

pseudo_r2 = 1 - (ll_model / ll_null)

print(f"McFadden's Pseudo R² : {pseudo_r2:.4f}")


# =====================================================
# MODEL 2 : RANDOM FOREST
# TRUE NESTED CROSS VALIDATION
# =====================================================

import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_score
)
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)

from optuna.visualization.matplotlib import (
    plot_optimization_history
)

print("\n" + "="*60)
print("MODEL 2 : RANDOM FOREST")
print("="*60)

# =====================================================
# MODEL 2 : RANDOM FOREST
# TRUE NESTED CROSS VALIDATION
# =====================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

outer_auc_scores = []
outer_train_scores = []

best_params_each_fold = []
studies = []
trained_models = []

# =====================================================
# OUTER LOOP
# =====================================================

for fold, (outer_train_idx, outer_valid_idx) in enumerate(
    outer_cv.split(X_train, y_train),
    start=1
):

    print()
    print("=" * 60)
    print(f"OUTER FOLD {fold}")
    print("=" * 60)

    X_outer_train = X_train.iloc[outer_train_idx]
    X_outer_valid = X_train.iloc[outer_valid_idx]

    y_outer_train = y_train.iloc[outer_train_idx]
    y_outer_valid = y_train.iloc[outer_valid_idx]

    # -----------------------------
    # OPTUNA OBJECTIVE
    # -----------------------------

    def objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                1000
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                30
            ),

            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10
            ),

            "max_features": trial.suggest_categorical(
                "max_features",
                [
                    "sqrt",
                    "log2",
                    None
                ]
            ),

            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1

        }

        pipeline = Pipeline([

            (
                "imputer",
                KNNImputer(
                    n_neighbors=5,
                    weights="distance"
                )
            ),

            (
                "rf",
                RandomForestClassifier(**params)
            )

        ])

        score = cross_val_score(
            estimator=pipeline,
            X=X_outer_train,
            y=y_outer_train,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        return score

    # -----------------------------
    # OPTUNA
    # -----------------------------

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True
    )

    studies.append(study)

    print()
    print("Best Parameters")
    print(study.best_params)

    print()
    print(f"Best Inner CV AUC : {study.best_value:.4f}")

    # -----------------------------
    # TRAIN BEST MODEL
    # -----------------------------

    best_pipeline = Pipeline([

        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                **study.best_params,
                random_state=42,
                n_jobs=-1
            )
        )

    ])

    best_pipeline.fit(
        X_outer_train,
        y_outer_train
    )

    # -----------------------------
    # TRAIN PERFORMANCE
    # -----------------------------

    train_probability = best_pipeline.predict_proba(
        X_outer_train
    )[:, 1]

    train_auc = roc_auc_score(
        y_outer_train,
        train_probability
    )

    # -----------------------------
    # VALIDATION PERFORMANCE
    # -----------------------------

    validation_probability = best_pipeline.predict_proba(
        X_outer_valid
    )[:, 1]

    validation_auc = roc_auc_score(
        y_outer_valid,
        validation_probability
    )

    # -----------------------------
    # STORE RESULTS
    # -----------------------------

    outer_train_scores.append(train_auc)
    outer_auc_scores.append(validation_auc)

    best_params_each_fold.append(study.best_params)
    trained_models.append(best_pipeline)

    print()
    print(f"Training AUC   : {train_auc:.4f}")
    print(f"Validation AUC : {validation_auc:.4f}")

# =====================================================
# END OF OUTER LOOP
# =====================================================

# =====================================================
# TRUE NESTED CV RESULTS
# =====================================================

print()
print("="*60)
print("TRUE NESTED CROSS VALIDATION RESULTS")
print("="*60)

for i, score in enumerate(outer_auc_scores, start=1):

    print(f"Fold {i}: {score:.4f}")

print()

print(f"Mean Validation AUC : {np.mean(outer_auc_scores):.4f}")
print(f"Std Validation AUC  : {np.std(outer_auc_scores):.4f}")

print()

print(f"Mean Training AUC : {np.mean(outer_train_scores):.4f}")
print(f"Std Training AUC  : {np.std(outer_train_scores):.4f}")

print()

print(
    f"Average Overfitting Gap : "
    f"{np.mean(outer_train_scores)-np.mean(outer_auc_scores):.4f}"
)

#Best params every fold 
print()

print("="*60)
print("BEST PARAMETERS FROM EACH OUTER FOLD")
print("="*60)

for i, params in enumerate(best_params_each_fold, start=1):

    print()

    print(f"Fold {i}")

    print(params)
    
#FINAL OBJECTIVE YAY!
print()
print("="*60)
print("FINAL MODEL TRAINING")
print("="*60)

def final_objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            30
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        ),

        "max_features": trial.suggest_categorical(
            "max_features",
            [
                "sqrt",
                "log2",
                None
            ]
        ),

        "bootstrap": True,

        "class_weight": "balanced",

        "random_state": 42,

        "n_jobs": -1

    }

    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("rf", RandomForestClassifier(**params))
    ])

    score = cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()

    return score    

#FINAL OPTUNA
final_study = optuna.create_study(direction="maximize")
final_study.optimize(final_objective,n_trials=50,show_progress_bar=True)

print()
print("FINAL BEST PARAMETERS")
print(final_study.best_params)
print()
print(f"Final CV AUC : {final_study.best_value:.4f}")

#FINAL RF PIPELINE 
rf_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ("rf", RandomForestClassifier(
        **final_study.best_params,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

#Test evaluation 
rf_pred = rf_pipeline.predict(X_test)
rf_prob = rf_pipeline.predict_proba(X_test)[:,1]
print()
print("="*60)
print("FINAL TEST PERFORMANCE")
print("="*60)
print(f"Test ROC AUC : {roc_auc_score(y_test, rf_prob):.4f}")
print(f"Brier Score  : {brier_score_loss(y_test, rf_prob):.4f}")

# =====================================================
# MODEL 3 : XGBOOST
# TRUE NESTED CROSS VALIDATION
# =====================================================

from xgboost import XGBClassifier

print("\n" + "="*60)
print("MODEL 3 : XGBOOST")
print("="*60)

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

outer_auc_scores = []
outer_train_scores = []

best_params_each_fold = []
studies = []
trained_models = []

# =====================================================
# OUTER LOOP
# =====================================================

for fold, (outer_train_idx, outer_valid_idx) in enumerate(
    outer_cv.split(X_train, y_train),
    start=1
):

    print()
    print("=" * 60)
    print(f"OUTER FOLD {fold}")
    print("=" * 60)

    X_outer_train = X_train.iloc[outer_train_idx]
    X_outer_valid = X_train.iloc[outer_valid_idx]

    y_outer_train = y_train.iloc[outer_train_idx]
    y_outer_valid = y_train.iloc[outer_valid_idx]

    # -------------------------------------------------
    # OPTUNA OBJECTIVE
    # -------------------------------------------------

    def xgb_objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                1000
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0,
                5
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10,
                log=True
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10,
                log=True
            ),

            "objective": "binary:logistic",

            "eval_metric": "auc",

            "random_state": 42,

            "n_jobs": -1

        }

        pipeline = Pipeline([

            (
                "imputer",
                KNNImputer(
                    n_neighbors=5,
                    weights="distance"
                )
            ),

            (
                "xgb",
                XGBClassifier(**params)
            )

        ])

        score = cross_val_score(

            estimator=pipeline,

            X=X_outer_train,

            y=y_outer_train,

            cv=inner_cv,

            scoring="roc_auc",

            n_jobs=-1

        ).mean()

        return score

    # -------------------------------------------------
    # OPTUNA
    # -------------------------------------------------

    study = optuna.create_study(direction="maximize")

    study.optimize(
        xgb_objective,
        n_trials=50,
        show_progress_bar=True
    )

    studies.append(study)

    print()
    print("Best Parameters")
    print(study.best_params)

    print()
    print(f"Best Inner CV AUC : {study.best_value:.4f}")

    # -------------------------------------------------
    # TRAIN BEST MODEL
    # -------------------------------------------------

    best_pipeline = Pipeline([

        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),

        (
            "xgb",
            XGBClassifier(

                **study.best_params,

                objective="binary:logistic",

                eval_metric="auc",

                random_state=42,

                n_jobs=-1

            )
        )

    ])

    best_pipeline.fit(
        X_outer_train,
        y_outer_train
    )

    # -------------------------------------------------
    # TRAIN PERFORMANCE
    # -------------------------------------------------

    train_probability = best_pipeline.predict_proba(
        X_outer_train
    )[:, 1]

    train_auc = roc_auc_score(
        y_outer_train,
        train_probability
    )

    # -------------------------------------------------
    # VALIDATION PERFORMANCE
    # -------------------------------------------------

    validation_probability = best_pipeline.predict_proba(
        X_outer_valid
    )[:, 1]

    validation_auc = roc_auc_score(
        y_outer_valid,
        validation_probability
    )

    # -------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------

    outer_train_scores.append(train_auc)
    outer_auc_scores.append(validation_auc)

    best_params_each_fold.append(study.best_params)
    trained_models.append(best_pipeline)

    print()
    print(f"Training AUC   : {train_auc:.4f}")
    print(f"Validation AUC : {validation_auc:.4f}")

# =====================================================
# END OF OUTER LOOP
# =====================================================

print()
print("=" * 60)
print("TRUE NESTED CROSS VALIDATION RESULTS")
print("=" * 60)

for i, score in enumerate(outer_auc_scores, start=1):

    print(f"Fold {i}: {score:.4f}")

print()

print(f"Mean Validation AUC : {np.mean(outer_auc_scores):.4f}")
print(f"Std Validation AUC  : {np.std(outer_auc_scores):.4f}")

print()

print(f"Mean Training AUC : {np.mean(outer_train_scores):.4f}")
print(f"Std Training AUC  : {np.std(outer_train_scores):.4f}")

print()

print(
    f"Average Overfitting Gap : "
    f"{np.mean(outer_train_scores) - np.mean(outer_auc_scores):.4f}"
)

print()
print("=" * 60)
print("BEST PARAMETERS FROM EACH OUTER FOLD")
print("=" * 60)

for i, params in enumerate(best_params_each_fold, start=1):

    print()
    print(f"Fold {i}")
    print(params)

# =====================================================
# FINAL MODEL TRAINING
# =====================================================

print()
print("=" * 60)
print("FINAL MODEL TRAINING")
print("=" * 60)

def final_xgb_objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0,
            5
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10,
            log=True
        ),

        "objective": "binary:logistic",

        "eval_metric": "auc",

        "random_state": 42,

        "n_jobs": -1

    }

    pipeline = Pipeline([

        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),

        (
            "xgb",
            XGBClassifier(**params)
        )

    ])

    score = cross_val_score(

        estimator=pipeline,

        X=X_train,

        y=y_train,

        cv=inner_cv,

        scoring="roc_auc",

        n_jobs=-1

    ).mean()

    return score


# =====================================================
# FINAL OPTUNA
# =====================================================

xgb_final_study = optuna.create_study(
    direction="maximize"
)

xgb_final_study.optimize(
    final_xgb_objective,
    n_trials=50,
    show_progress_bar=True
)

print()
print("FINAL BEST PARAMETERS")
print(xgb_final_study.best_params)

print()
print(f"Final CV AUC : {xgb_final_study.best_value:.4f}")


# =====================================================
# FINAL XGBOOST PIPELINE
# =====================================================

xgb_pipeline = Pipeline([

    (
        "imputer",
        KNNImputer(
            n_neighbors=5,
            weights="distance"
        )
    ),

    (
        "xgb",
        XGBClassifier(

            **xgb_final_study.best_params,

            objective="binary:logistic",

            eval_metric="auc",

            random_state=42,

            n_jobs=-1

        )
    )

])

xgb_pipeline.fit(
    X_train,
    y_train
)


# =====================================================
# TEST SET EVALUATION
# =====================================================

xgb_pred = xgb_pipeline.predict(
    X_test
)

xgb_prob = xgb_pipeline.predict_proba(
    X_test
)[:, 1]

print()
print("=" * 60)
print("FINAL TEST PERFORMANCE")
print("=" * 60)

print(
    f"Test ROC AUC : "
    f"{roc_auc_score(y_test, xgb_prob):.4f}"
)

print(
    f"Brier Score  : "
    f"{brier_score_loss(y_test, xgb_prob):.4f}"
)


# =====================================================
# OPTUNA OPTIMIZATION HISTORY
# =====================================================

plt.figure(figsize=(8,5))

plot_optimization_history(
    xgb_final_study
)

plt.title("XGBoost Optuna Optimization History")

plt.tight_layout()

plt.show()



# =====================================================
# MODEL 4 : LIGHTGBM
# TRUE NESTED CROSS VALIDATION
# =====================================================

from lightgbm import LGBMClassifier

print("\n" + "="*60)
print("MODEL 4 : LIGHTGBM")
print("="*60)

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

outer_auc_scores = []
outer_train_scores = []

best_params_each_fold = []
studies = []
trained_models = []

# =====================================================
# OUTER LOOP
# =====================================================

for fold, (outer_train_idx, outer_valid_idx) in enumerate(
    outer_cv.split(X_train, y_train),
    start=1
):

    print()
    print("=" * 60)
    print(f"OUTER FOLD {fold}")
    print("=" * 60)

    X_outer_train = X_train.iloc[outer_train_idx]
    X_outer_valid = X_train.iloc[outer_valid_idx]

    y_outer_train = y_train.iloc[outer_train_idx]
    y_outer_valid = y_train.iloc[outer_valid_idx]

    # -------------------------------------------------
    # OPTUNA OBJECTIVE
    # -------------------------------------------------

    def lgbm_objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                1000
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                -1,
                15
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                255
            ),

            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                5,
                100
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10,
                log=True
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10,
                log=True
            ),

            "objective": "binary",

            "random_state": 42,

            "verbosity": -1,

            "n_jobs": -1

        }

        pipeline = Pipeline([

            (
                "imputer",
                KNNImputer(
                    n_neighbors=5,
                    weights="distance"
                )
            ),

            (
                "lgbm",
                LGBMClassifier(**params)
            )

        ])

        score = cross_val_score(

            estimator=pipeline,

            X=X_outer_train,

            y=y_outer_train,

            cv=inner_cv,

            scoring="roc_auc",

            n_jobs=-1

        ).mean()

        return score

    # -------------------------------------------------
    # OPTUNA
    # -------------------------------------------------

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lgbm_objective,
        n_trials=50,
        show_progress_bar=True
    )

    studies.append(study)

    print()
    print("Best Parameters")
    print(study.best_params)

    print()
    print(f"Best Inner CV AUC : {study.best_value:.4f}")

    # -------------------------------------------------
    # TRAIN BEST MODEL
    # -------------------------------------------------

    best_pipeline = Pipeline([

        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),

        (
            "lgbm",
            LGBMClassifier(

                **study.best_params,

                objective="binary",

                random_state=42,

                verbosity=-1,

                n_jobs=-1

            )
        )

    ])

    best_pipeline.fit(
        X_outer_train,
        y_outer_train
    )

    # -------------------------------------------------
    # TRAIN PERFORMANCE
    # -------------------------------------------------

    train_probability = best_pipeline.predict_proba(
        X_outer_train
    )[:, 1]

    train_auc = roc_auc_score(
        y_outer_train,
        train_probability
    )

    # -------------------------------------------------
    # VALIDATION PERFORMANCE
    # -------------------------------------------------

    validation_probability = best_pipeline.predict_proba(
        X_outer_valid
    )[:, 1]

    validation_auc = roc_auc_score(
        y_outer_valid,
        validation_probability
    )

    # -------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------

    outer_train_scores.append(train_auc)
    outer_auc_scores.append(validation_auc)

    best_params_each_fold.append(study.best_params)
    trained_models.append(best_pipeline)

    print()
    print(f"Training AUC   : {train_auc:.4f}")
    print(f"Validation AUC : {validation_auc:.4f}")

# =====================================================
# END OF OUTER LOOP
# =====================================================

print()
print("="*60)
print("TRUE NESTED CROSS VALIDATION RESULTS")
print("="*60)

for i, score in enumerate(outer_auc_scores, start=1):

    print(f"Fold {i}: {score:.4f}")

print()

print(f"Mean Validation AUC : {np.mean(outer_auc_scores):.4f}")
print(f"Std Validation AUC  : {np.std(outer_auc_scores):.4f}")

print()

print(f"Mean Training AUC : {np.mean(outer_train_scores):.4f}")
print(f"Std Training AUC  : {np.std(outer_train_scores):.4f}")

print()

print(
    f"Average Overfitting Gap : "
    f"{np.mean(outer_train_scores)-np.mean(outer_auc_scores):.4f}"
)

print()

print("="*60)
print("BEST PARAMETERS FROM EACH OUTER FOLD")
print("="*60)

for i, params in enumerate(best_params_each_fold, start=1):

    print()
    print(f"Fold {i}")
    print(params)


# =====================================================
# FINAL MODEL TRAINING
# =====================================================

print()
print("=" * 60)
print("FINAL MODEL TRAINING")
print("=" * 60)

def final_lgbm_objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            -1,
            15
        ),

        "num_leaves": trial.suggest_int(
            "num_leaves",
            15,
            255
        ),
 
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            5,
            100
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10,
            log=True
        ),

        "objective": "binary",

        "random_state": 42,

        "verbosity": -1,

        "n_jobs": -1

    }

    pipeline = Pipeline([

        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),

        (
            "lgbm",
            LGBMClassifier(**params)
        )

    ])

    score = cross_val_score(

        estimator=pipeline,

        X=X_train,

        y=y_train,

        cv=inner_cv,

        scoring="roc_auc",

        n_jobs=-1

    ).mean()

    return score


# =====================================================
# FINAL OPTUNA
# =====================================================

lgbm_final_study = optuna.create_study(
    direction="maximize"
)

lgbm_final_study.optimize(
    final_lgbm_objective,
    n_trials=50,
    show_progress_bar=True
)

print()
print("FINAL BEST PARAMETERS")
print(lgbm_final_study.best_params)

print()
print(f"Final CV AUC : {lgbm_final_study.best_value:.4f}")


# =====================================================
# FINAL LIGHTGBM PIPELINE
# =====================================================

lgbm_pipeline = Pipeline([

    (
        "imputer",
        KNNImputer(
            n_neighbors=5,
            weights="distance"
        )
    ),

    (
        "lgbm",
        LGBMClassifier(

            **lgbm_final_study.best_params,

            objective="binary",

            random_state=42,

            verbosity=-1,

            n_jobs=-1

        )
    )

])

lgbm_pipeline.fit(
    X_train,
    y_train
)


# =====================================================
# TEST SET EVALUATION
# =====================================================

lgbm_pred = lgbm_pipeline.predict(
    X_test
)

lgbm_prob = lgbm_pipeline.predict_proba(
    X_test
)[:, 1]

print()
print("=" * 60)
print("FINAL TEST PERFORMANCE")
print("=" * 60)

print(
    f"Test ROC AUC : "
    f"{roc_auc_score(y_test, lgbm_prob):.4f}"
)

print(
    f"Brier Score  : "
    f"{brier_score_loss(y_test, lgbm_prob):.4f}"
)


# =====================================================
# OPTUNA OPTIMIZATION HISTORY
# =====================================================

plt.figure(figsize=(8,5))

plot_optimization_history(
    lgbm_final_study
)

plt.title("LightGBM Optuna Optimization History")

plt.tight_layout()

plt.show()

# =====================================
# PART 3 : Model Validation 
# =====================================
# Now take the best model(s) and generate:
# ROC curves (all models on one plot)
# Precision–Recall curves
# Calibration curves
# Confusion matrices
# KS statistic
# Gini coefficient
# Decile/Lift chart


# MODEL COMPARISON
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Random Forest",

        "XGBoost",

        "LightGBM"

    ],

    "ROC AUC": [

        roc_auc_score(y_test, y_prob),

        roc_auc_score(y_test, rf_prob),

        roc_auc_score(y_test, xgb_prob),

        roc_auc_score(y_test, lgbm_prob)

    ],

    "Brier Score": [

        brier_score_loss(y_test, y_prob),

        brier_score_loss(y_test, rf_prob),

        brier_score_loss(y_test, xgb_prob),

        brier_score_loss(y_test, lgbm_prob)

    ]

})

print(comparison.round(4))


#ROC CURVES

from sklearn.metrics import roc_curve

plt.figure(figsize=(8,6))

models = {

    "Logistic Regression": y_prob,

    "Random Forest": rf_prob,

    "XGBoost": xgb_prob,

    "LightGBM": lgbm_prob

}

for name, probability in models.items():

    fpr, tpr, _ = roc_curve(

        y_test,

        probability

    )

    auc = roc_auc_score(

        y_test,

        probability

    )

    plt.plot(

        fpr,

        tpr,

        label=f"{name} (AUC = {auc:.3f})"

    )

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend()

plt.grid(True)

plt.show()

# PRECISION-RECALL CURVES
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

plt.figure(figsize=(8,6))

for name, probability in models.items():

    precision, recall, _ = precision_recall_curve(

        y_test,

        probability

    )

    ap = average_precision_score(

        y_test,

        probability

    )

    plt.plot(

        recall,

        precision,

        label=f"{name} (AP = {ap:.3f})"

    )

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curves")

plt.legend()

plt.grid(True)

plt.show()

# CALIRATION CURVES
from sklearn.calibration import calibration_curve

plt.figure(figsize=(8,6))

for name, probability in models.items():

    frac_pos, mean_pred = calibration_curve(

        y_test,

        probability,

        n_bins=10

    )

    plt.plot(

        mean_pred,

        frac_pos,

        marker="o",

        label=name

    )

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)

plt.xlabel("Mean Predicted Probability")

plt.ylabel("Observed Frequency")

plt.title("Calibration Curves")

plt.legend()

plt.grid(True)

plt.show()


# Confusion matrices
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)

predictions = {

    "Logistic Regression": y_prob,

    "Random Forest": rf_prob,

    "XGBoost": xgb_prob,

    "LightGBM": lgbm_prob

}

for name, probability in predictions.items():

    prediction = (probability >= 0.50).astype(int)

    cm = confusion_matrix(
        y_test,
        prediction
    )

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    print(cm)

    print(f"Accuracy  : {accuracy_score(y_test,prediction):.4f}")
    print(f"Precision : {precision_score(y_test,prediction):.4f}")
    print(f"Recall    : {recall_score(y_test,prediction):.4f}")
    print(f"F1 Score  : {f1_score(y_test,prediction):.4f}")

    ConfusionMatrixDisplay(cm).plot(
        cmap="Blues"
    )

    plt.title(name)

    plt.show()

#KS 
from scipy.stats import ks_2samp
print("\n" + "=" * 60)
print("KOLMOGOROV-SMIRNOV STATISTIC")
print("=" * 60)

for name, probability in predictions.items():

    default_scores = probability[y_test == 1]

    non_default_scores = probability[y_test == 0]

    ks = ks_2samp(

        default_scores,

        non_default_scores

    )

    print(f"{name:22s} KS = {ks.statistic:.4f}")

plt.figure(figsize=(8,6))

for name, probability in predictions.items():

    temp = pd.DataFrame({

        "Probability": probability,

        "Default": y_test.values

    })

    temp = temp.sort_values(
        "Probability",
        ascending=False
    )

    temp["Bad Cum"] = (

        temp["Default"].cumsum()

        / temp["Default"].sum()

    )

    temp["Good Cum"] = (

        ((1-temp["Default"]).cumsum())

        / ((1-temp["Default"]).sum())

    )

    ks_curve = np.abs(

        temp["Bad Cum"]

        - temp["Good Cum"]

    )

    plt.plot(

        ks_curve,

        label=f"{name} (KS={ks_curve.max():.3f})"

    )

plt.legend()

plt.title("Kolmogorov-Smirnov Curves")

plt.ylabel("Separation")

plt.xlabel("Ordered Borrowers")

plt.grid(True)

plt.show()

#GİNİ
print("\n" + "=" * 60)
print("GINI COEFFICIENT")
print("=" * 60)

for name, probability in predictions.items():

    auc = roc_auc_score(

        y_test,

        probability

    )

    gini = 2 * auc - 1

    print(f"{name:22s} {gini:.4f}")




# DECILE ANALYSIS
def decile_table(y_true, probability):

    table = pd.DataFrame(
        {
            "Actual": y_true,
            "Probability": probability
        }
    )

    table = table.sort_values(
        "Probability",
        ascending=False
    )

    table["Decile"] = (
        pd.qcut(
            table.index,
            10,
            labels=False
        ) + 1
    )

    summary = table.groupby("Decile").agg(
        Accounts=("Actual", "count"),
        Defaults=("Actual", "sum"),
        Mean_PD=("Probability", "mean")
    )

    summary["Default Rate"] = (
        summary["Defaults"]
        / summary["Accounts"]
    )

    summary["Cumulative Defaults"] = (
        summary["Defaults"].cumsum()
    )

    summary["Cumulative Accounts"] = (
        summary["Accounts"].cumsum()
    )

    return summary

for name, probability in predictions.items():

    print()

    print("=" * 60)

    print(name)

    print("=" * 60)

    print(

        decile_table(

            y_test,

            probability

        ).round(4)

    )

#LIFT
plt.figure(figsize=(8,6))

overall_rate = y_test.mean()

for name, probability in predictions.items():

    summary = decile_table(

        y_test,

        probability

    )

    lift = (

        summary["Default Rate"]

        / overall_rate

    )

    plt.plot(

        range(1,11),

        lift,

        marker="o",

        label=name

    )

plt.xlabel("Decile")

plt.ylabel("Lift")

plt.title("Lift Chart")

plt.legend()

plt.grid(True)

plt.show()

#GAINS CHART
plt.figure(figsize=(8,6))

for name, probability in predictions.items():

    summary = decile_table(

        y_test,

        probability

    )

    gains = (

        summary["Defaults"].cumsum()

        / summary["Defaults"].sum()

    )

    plt.plot(

        range(1,11),

        gains,

        marker="o",

        label=name

    )

plt.plot(

    range(1,11),

    np.linspace(0.1,1,10),

    "--",

    color="black",

    label="Random"

)

plt.xlabel("Decile")

plt.ylabel("Cumulative Default Capture")

plt.title("Cumulative Gains Chart")

plt.legend()

plt.grid(True)

plt.show()


# =====================================
# Part 6 : Interpretability
# =====================================

# Here the method depends on the model.

# For Logistic Regression:

# Coefficients
# Odds ratios

# For Random Forest:

# Feature importance
# Permutation importance (preferred over impurity-based importance)

# For XGBoost/LightGBM:

# SHAP values
# SHAP summary plot
# SHAP dependence plots (optional)

# Model Interpretation Summary
#       • Most influential variables
#       • Agreement across models
#       • Credit risk interpretation

print("\n" + "="*60)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("="*60)

logistic_model = pipeline.named_steps["logistic"]

coef_table = pd.DataFrame({

    "Feature": X_train.columns,

    "Coefficient": logistic_model.coef_[0]

})

coef_table["Odds Ratio"] = np.exp(
    coef_table["Coefficient"]
)

coef_table = coef_table.sort_values(
    "Coefficient",
    key=np.abs,
    ascending=False
)

print(coef_table.round(4))

plt.figure(figsize=(10,7))

coef_table_sorted = coef_table.sort_values(
    "Coefficient"
)

plt.barh(
    coef_table_sorted["Feature"],
    coef_table_sorted["Coefficient"]
)

plt.title("Logistic Regression Coefficients")

plt.xlabel("Coefficient")

plt.grid(True)

plt.tight_layout()

plt.show()

#Random Forest Impurity importance
print("\n" + "="*60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("="*60)

rf_model = rf_pipeline.named_steps["rf"]

rf_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": rf_model.feature_importances_

})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
)

print(rf_importance.round(4))

plt.figure(figsize=(9,6))

plt.barh(

    rf_importance["Feature"],

    rf_importance["Importance"]

)

plt.gca().invert_yaxis()

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.show()


#Permutation Importance (preferred)
from sklearn.inspection import permutation_importance

result = permutation_importance(

    rf_pipeline,

    X_test,

    y_test,

    scoring="roc_auc",

    n_repeats=20,

    random_state=42,

    n_jobs=-1

)

perm = pd.DataFrame({

    "Feature": X_test.columns,

    "Importance": result.importances_mean

})

perm = perm.sort_values(

    "Importance",

    ascending=False

)

print(perm.round(4))


plt.figure(figsize=(9,6))

plt.barh(

    perm["Feature"],

    perm["Importance"]

)

plt.gca().invert_yaxis()

plt.title("Permutation Importance")

plt.tight_layout()

plt.show()


# SHAP INTERPRETABILITY
# XGBoost & LightGBM


import shap

models = {

    "XGBoost": xgb_pipeline,

    "LightGBM": lgbm_pipeline

}

for name, pipeline in models.items():

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    model = pipeline.named_steps[
        "xgb" if name == "XGBoost" else "lgbm"
    ]

    X_test_imputed = pipeline.named_steps["imputer"].transform(
        X_test
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test_imputed
    )

    # -----------------------------------
    # SHAP Summary Plot
    # -----------------------------------

    shap.summary_plot(

        shap_values,

        X_test_imputed,

        feature_names=X_test.columns,

        show=True

    )

    # -----------------------------------
    # SHAP Feature Importance
    # -----------------------------------

    shap.summary_plot(

        shap_values,

        X_test_imputed,

        feature_names=X_test.columns,

        plot_type="bar",

        show=True

    )

    # -----------------------------------
    # SHAP Dependence Plot
    # -----------------------------------

    shap.dependence_plot(

        "kkb_score",

        shap_values,

        X_test_imputed,

        feature_names=X_test.columns

    )

print("\n" + "="*60)
print("MODEL INTERPRETATION SUMMARY")
print("="*60)

print("""
• Logistic Regression provides transparent coefficients and odds ratios,
  indicating the direction and magnitude of each variable's effect.

• Random Forest permutation importance measures the reduction in predictive
  performance when each feature is randomly shuffled, providing a robust
  estimate of variable importance.

• SHAP values explain individual predictions for XGBoost and LightGBM,
  quantifying each feature's contribution to the predicted probability of
  default.

• Agreement across multiple interpretation methods increases confidence
  that the identified risk drivers are genuine rather than model-specific.
""")

# =====================================
# Part 7 : Robustness
# =====================================
# Since it's explainable our champion in a realistic case for reporting purposes will be logistic regression however since this is a ML project my challenger champ is LightGBM so let's do the robustness analysis..

# Bootstrap confidence interval for AUC
# Sensitivity to threshold
# Learning curve 
# Calibration assessment

# =====================================================
# BOOTSTRAP CONFIDENCE INTERVAL FOR ROC AUC
# =====================================================

from sklearn.utils import resample

print("\n" + "="*60)
print("BOOTSTRAP CONFIDENCE INTERVAL FOR ROC AUC")
print("="*60)

bootstrap_iterations = 1000

models = {

    "Logistic Regression": y_prob,

    "LightGBM": lgbm_prob

}

for name, probability in models.items():

    auc_scores = []

    for i in range(bootstrap_iterations):

        indices = resample(

            np.arange(len(y_test)),
            replace=True,
            random_state=i

        )

        if len(np.unique(y_test.iloc[indices])) < 2:
            continue

        auc_scores.append(

            roc_auc_score(

                y_test.iloc[indices],
                probability[indices]

            )

        )

    lower = np.percentile(auc_scores,2.5)
    upper = np.percentile(auc_scores,97.5)

    print(f"\n{name}")

    print(f"Mean AUC : {np.mean(auc_scores):.4f}")
    print(f"95% CI   : [{lower:.4f}, {upper:.4f}]")

# =====================================================
# THRESHOLD SENSITIVITY
# =====================================================

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("\n" + "="*60)
print("THRESHOLD SENSITIVITY")
print("="*60)

thresholds = np.arange(0.10,0.91,0.10)

for name, probability in models.items():

    print()
    print("="*50)
    print(name)
    print("="*50)

    results = []

    for threshold in thresholds:

        prediction = (probability >= threshold).astype(int)

        results.append({

            "Threshold": threshold,

            "Precision": precision_score(
                y_test,
                prediction,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                prediction,
                zero_division=0
            ),

            "F1": f1_score(
                y_test,
                prediction,
                zero_division=0
            )

        })

    threshold_table = pd.DataFrame(results)

    print(threshold_table.round(3))

    plt.figure(figsize=(8,5))

    plt.plot(
        threshold_table["Threshold"],
        threshold_table["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_table["Threshold"],
        threshold_table["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_table["Threshold"],
        threshold_table["F1"],
        marker="o",
        label="F1"
    )

    plt.xlabel("Classification Threshold")

    plt.ylabel("Metric")

    plt.title(name)

    plt.grid(True)

    plt.legend()

    plt.show()

# =====================================================
# LEARNING CURVES
# =====================================================

from sklearn.model_selection import learning_curve

print("\n" + "="*60)
print("LEARNING CURVES")
print("="*60)

estimators = {

    "Logistic Regression": pipeline,

    "LightGBM": lgbm_pipeline

}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, estimator in estimators.items():

    train_sizes, train_scores, validation_scores = learning_curve(

        estimator,

        X_train,

        y_train,

        cv=cv,

        scoring="roc_auc",

        train_sizes=np.linspace(0.1,1.0,10),

        n_jobs=-1

    )

    plt.figure(figsize=(8,5))

    plt.plot(

        train_sizes,

        train_scores.mean(axis=1),

        marker="o",

        label="Training"

    )

    plt.plot(

        train_sizes,

        validation_scores.mean(axis=1),

        marker="o",

        label="Validation"

    )

    plt.xlabel("Training Samples")

    plt.ylabel("ROC AUC")

    plt.title(f"Learning Curve - {name}")

    plt.grid(True)

    plt.legend()

    plt.show()

# =====================================================
# EXPECTED CALIBRATION ERROR (ECE)
# =====================================================

print("\n" + "="*60)
print("EXPECTED CALIBRATION ERROR")
print("="*60)

def expected_calibration_error(

    y_true,

    probability,

    bins=10

):

    edges = np.linspace(0,1,bins+1)

    ece = 0

    for i in range(bins):

        mask = (

            (probability >= edges[i])

            &

            (probability < edges[i+1])

        )

        if np.sum(mask)==0:
            continue

        accuracy = y_true[mask].mean()

        confidence = probability[mask].mean()

        ece += (

            np.abs(accuracy-confidence)

            * np.sum(mask)

            / len(y_true)

        )

    return ece

for name, probability in models.items():

    ece = expected_calibration_error(

        y_test.values,

        probability

    )

    print(f"{name:22s} ECE = {ece:.4f}")



# =====================================================
# EXPECTED LOSS CALCULATION
# =====================================================

def calculate_expected_loss(
    borrower_features,
    loan_amount,
    model,
    recovery_rate=0.10
):

    # ---------------------------------------------
    # Convert borrower information to DataFrame
    # ---------------------------------------------

    if isinstance(borrower_features, dict):
        features_df = pd.DataFrame([borrower_features])
    else:
        features_df = borrower_features.copy()

    # ---------------------------------------------
    # Input Validation
    # ---------------------------------------------

    if loan_amount <= 0:
        raise ValueError("Loan amount must be positive.")

    if not (0 <= recovery_rate <= 1):
        raise ValueError("Recovery rate must be between 0 and 1.")

    # ---------------------------------------------
    # SAME FEATURE ENGINEERING AS TRAINING
    # ---------------------------------------------

    features_df["dti"] = (
        features_df["total_debt_o"]
        / features_df["income"]
    )

    features_df["loan_income_ratio"] = (
        features_df["loan_amount_o"]
        / features_df["income"]
    )

    features_df["debt_per_credit_line"] = (
        features_df["total_debt_o"]
        / features_df["credit_lines_outstanding"]
    )

    features_df["loan_share"] = (
        features_df["loan_amount_o"]
        / features_df["total_debt_o"]
    )

    features_df["income_per_credit_line"] = (
        features_df["income"]
        / features_df["credit_lines_outstanding"]
    )

    features_df["employment_group"] = pd.cut(

        features_df["years_employed"],

        bins=[0,2,5,10,100],

        labels=[
            "New",
            "Experienced",
            "Stable",
            "Veteran"
        ]

    )

    # ---------------------------------------------
    # Handle infinite values
    # ---------------------------------------------

    features_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # ---------------------------------------------
    # One-Hot Encoding
    # ---------------------------------------------

    features_df = pd.get_dummies(

        features_df,

        columns=["employment_group"],

        drop_first=True

    )

    # ---------------------------------------------
    # Match training columns
    # ---------------------------------------------

    features_df = features_df.reindex(

        columns=X_train.columns,

        fill_value=0

    )

    # ---------------------------------------------
    # Probability of Default
    # ---------------------------------------------

    PD = model.predict_proba(
        features_df
    )[0,1]

    # ---------------------------------------------
    # Expected Loss Components
    # ---------------------------------------------

    LGD = 1 - recovery_rate

    EAD = loan_amount

    EL = PD * LGD * EAD

    return {

        "PD": PD,

        "LGD": LGD,

        "EAD": EAD,

        "EL": EL

    }


# =====================================================
# EXAMPLE BORROWER
# =====================================================

new_borrower = {

    "credit_lines_outstanding":2,
    "loan_amount_o":49003,
    "total_debt_o":43865,
    "income":85000,
    "years_employed":1,
    "kkb_score":1850

}

specific_loan_amount = 66000


# =====================================================
# MODEL COMPARISON
# =====================================================

models = {

    "Logistic Regression": pipeline,

    "LightGBM": lgbm_pipeline

}

for name, model in models.items():

    results = calculate_expected_loss(

        borrower_features=new_borrower,

        loan_amount=specific_loan_amount,

        model=model

    )

    print()

    print("="*60)

    print(name)

    print("="*60)

    print(f"Probability of Default (PD) : {results['PD']:.2%}")

    print(f"Loss Given Default (LGD)    : {results['LGD']:.2%}")

    print(f"Exposure at Default (EAD)   : Rs{results['EAD']:,.2f}")

    print(f"Expected Loss (EL)          : Rs{results['EL']:,.2f}")


#############Credit Decision Engine (AN ALTERNATIVE SCORING APPROACH)
# =====================================================
# CREDIT DECISION ENGINE
# =====================================================

def credit_decision(results):

    PD = results["PD"]

    # ---------------------------------------------
    # Internal Risk Rating
    # ---------------------------------------------

    if PD < 0.01:
        rating = "AAA"

    elif PD < 0.03:
        rating = "AA"

    elif PD < 0.05:
        rating = "A"

    elif PD < 0.10:
        rating = "BBB"

    elif PD < 0.20:
        rating = "BB"

    elif PD < 0.40:
        rating = "B"

    else:
        rating = "CCC"

    # ---------------------------------------------
    # Credit Decision
    # ---------------------------------------------

    if PD < 0.05:
        decision = "APPROVE"

    elif PD < 0.15:
        decision = "MANUAL REVIEW"

    else:
        decision = "DECLINE"

    return {

        "Risk Rating": rating,

        "Decision": decision

    }

# =====================================================
# MODEL COMPARISON
# =====================================================

models = {

    "Logistic Regression": pipeline,

    "LightGBM": lgbm_pipeline

}

for name, model in models.items():

    results = calculate_expected_loss(

        borrower_features=new_borrower,

        loan_amount=specific_loan_amount,

        model=model

    )

    decision = credit_decision(results)

    print()

    print("="*60)
    print("CREDIT RISK REPORT")
    print("="*60)

    print(f"Model                         : {name}")
    print(f"Probability of Default (PD)   : {results['PD']:.2%}")
    print(f"Loss Given Default (LGD)      : {results['LGD']:.2%}")
    print(f"Exposure at Default (EAD)     : Rs{results['EAD']:,.2f}")
    print(f"Expected Loss (EL)            : Rs{results['EL']:,.2f}")
    print(f"Internal Risk Rating          : {decision['Risk Rating']}")
    print(f"Credit Decision               : {decision['Decision']}")

    print("="*60)
























