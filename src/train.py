"""
src/train.py  -  Step 4: Train and compare models
Run from the project root:  python src/train.py   (run src/preprocess.py first)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(__file__))
from features import FEATURE_COLUMNS, RAW_FEATURES

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------------- Load the data made by preprocess.py ----------------
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train[FEATURE_COLUMNS], train["fake"]
X_test, y_test = test[FEATURE_COLUMNS], test["fake"]

# 5-fold cross-validation: the training data is split into 5 parts; each part
# takes a turn as the "validation" set while the model learns from the other 4.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------- Model 1: Random Forest (tuned) ----------------
print("Tuning Random Forest with RandomizedSearchCV (5-fold CV)...")
rf_param_space = {
    "n_estimators": [100, 200, 300, 500],    # number of trees
    "max_depth": [None, 5, 8, 12, 16],       # how deep each tree may grow
    "min_samples_split": [2, 5, 10, 20],     # min rows needed to split a node
    "min_samples_leaf": [1, 2, 4, 8],        # min rows allowed in a leaf
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_space,
    n_iter=30, cv=cv, scoring="f1", random_state=42, n_jobs=-1,
)
rf_search.fit(X_train, y_train)
print("Best Random Forest settings:", rf_search.best_params_)

# ---------------- Model 2: Logistic Regression (baseline) ----------------
# Logistic regression is sensitive to feature size, so we scale first.
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
logreg.fit(X_train, y_train)

# ---------------- Model 3: Decision Tree (baseline, small grid) ----------------
dt_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 10]},
    cv=cv, scoring="f1", n_jobs=-1,
)
dt_search.fit(X_train, y_train)
print("Best Decision Tree settings:", dt_search.best_params_)

models = {
    "Random Forest": rf_search.best_estimator_,
    "Logistic Regression": logreg,
    "Decision Tree": dt_search.best_estimator_,
}

# ---------------- Compare all three ----------------
rows = []
for name, model in models.items():
    # CV F1 on the TRAINING data only. The test set stays untouched for choosing models.
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV F1 (mean)": cv_scores.mean(),
        "CV F1 (std)": cv_scores.std(),
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Precision": precision_score(y_test, pred),
        "Test Recall": recall_score(y_test, pred),
        "Test F1": f1_score(y_test, pred),
        "Test ROC-AUC": roc_auc_score(y_test, proba),
    })
table = pd.DataFrame(rows).round(4)
print("\n=== Model comparison ===")
print(table.to_string(index=False))
table.to_csv("reports/model_comparison.csv", index=False)

# ---------------- Pick the best model using CV score (NOT the test score) ----------------
best_name = table.sort_values("CV F1 (mean)", ascending=False).iloc[0]["Model"]
best_model = models[best_name]
print(f"\nBest model by cross-validation F1: {best_name}")

# ---------------- Save everything ----------------
joblib.dump(best_model, "models/best_model.joblib")
joblib.dump(models["Random Forest"], "models/random_forest.joblib")  # needed for importances
joblib.dump(FEATURE_COLUMNS, "models/feature_columns.joblib")

# Typical (median) raw values of REAL and FAKE training profiles.
# predict.py uses these for its "what-if" explanations.
reference = {
    "real": train[train.fake == 0][RAW_FEATURES].median().to_dict(),
    "fake": train[train.fake == 1][RAW_FEATURES].median().to_dict(),
}
with open("models/reference_profiles.json", "w") as f:
    json.dump(reference, f, indent=2)
with open("models/best_model_name.json", "w") as f:
    json.dump({"best_model": best_name, "rf_best_params": rf_search.best_params_}, f, indent=2)
print("Saved to models/: best_model, random_forest, feature_columns, reference_profiles")
