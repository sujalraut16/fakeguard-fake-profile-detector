"""
src/evaluate.py  -  Step 5: Evaluation + leakage / overfitting checks
Run from the project root:  python src/evaluate.py   (run src/train.py first)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve)

sys.path.append(os.path.dirname(__file__))
from features import FEATURE_COLUMNS

os.makedirs("reports", exist_ok=True)
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train[FEATURE_COLUMNS], train["fake"]
X_test, y_test = test[FEATURE_COLUMNS], test["fake"]

model = joblib.load("models/best_model.joblib")
rf = joblib.load("models/random_forest.joblib")
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# ================= 1. Main metrics =================
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}
print("=== Test-set metrics ===")
for k, v in metrics.items():
    print(f"  {k:<10}: {v:.4f}")

# ================= 2. Confusion matrix =================
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  FP = {fp} REAL accounts wrongly called FAKE")
print(f"  FN = {fn} FAKE accounts that slipped through as REAL")
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix (test set)")
plt.tight_layout(); plt.savefig("reports/05_confusion_matrix.png", dpi=150); plt.close()

# ================= 3. ROC curve =================
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, color="#d62828", lw=2, label=f"Model (AUC = {metrics['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curve (test set)"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("reports/06_roc_curve.png", dpi=150); plt.close()

# ================= 4. Feature importance (Random Forest) =================
impurity = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42,
                              scoring="f1", n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
impurity.plot.barh(ax=axes[0], color="#264653"); axes[0].set_title("Built-in importance (Random Forest)")
perm_s.plot.barh(ax=axes[1], color="#e9c46a");   axes[1].set_title("Permutation importance (test F1 drop)")
plt.tight_layout(); plt.savefig("reports/07_feature_importance.png", dpi=150); plt.close()
print("\nTop 6 features (built-in importance):")
print(impurity.sort_values(ascending=False).head(6).round(3).to_string())
print("\nTop 6 features (permutation importance, drop in test F1 when shuffled):")
print(perm_s.sort_values(ascending=False).head(6).round(3).to_string())

# ================= 5. Threshold table (precision vs recall trade-off) =================
print("\nEffect of the decision threshold (default is 0.50):")
print("  threshold  precision  recall  false_positives  false_negatives")
thr_rows = []
for t in [0.3, 0.5, 0.7, 0.9]:
    p = (proba >= t).astype(int)
    _, fp_t, fn_t, _ = confusion_matrix(y_test, p).ravel()
    thr_rows.append((t, precision_score(y_test, p), recall_score(y_test, p), fp_t, fn_t))
    print(f"  {t:<9}  {thr_rows[-1][1]:.3f}      {thr_rows[-1][2]:.3f}   {fp_t:<15}  {fn_t}")

# ================= 6. Leakage and overfitting checks =================
print("\n=== Leakage / overfitting checks ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Check A: any identical row in both train and test?
overlap = pd.merge(train.drop(columns="fake").drop_duplicates(),
                   test.drop(columns="fake").drop_duplicates(), how="inner").shape[0]
print(f"A) Rows appearing in BOTH train and test: {overlap}")

# Check B: train vs CV vs test accuracy (a big gap = overfitting)
train_acc = accuracy_score(y_train, model.predict(X_train))
cv_acc = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring="accuracy").mean()
print(f"B) Accuracy  train = {train_acc:.4f} | 5-fold CV = {cv_acc:.4f} | test = {metrics['accuracy']:.4f}")
print(f"   Train-minus-test gap = {train_acc - metrics['accuracy']:.4f}")

# Check C: shuffled labels. If the labels are random the model should score ~50%.
rng = np.random.RandomState(42)
y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
shuf_acc = cross_val_score(clone(model), X_train, y_shuffled, cv=cv, scoring="accuracy").mean()
print(f"C) CV accuracy with SHUFFLED labels = {shuf_acc:.4f}  (should be near 0.50)")

# Check D: how well does each SINGLE feature predict on its own?
#   (a lone feature scoring ~100% would suggest it secretly encodes the answer)
single = {}
for col in FEATURE_COLUMNS:
    stump = DecisionTreeClassifier(max_depth=3, random_state=42)
    single[col] = cross_val_score(stump, X_train[[col]], y_train, cv=cv, scoring="accuracy").mean()
single = pd.Series(single).sort_values(ascending=False)
print("D) Best single-feature accuracy (small tree, 5-fold CV):")
print(single.head(5).round(3).to_string())

# Check E: the simplest baseline
majority = max(y_test.mean(), 1 - y_test.mean())
print(f"E) Majority-class baseline accuracy = {majority:.3f}")

# Save numbers for the README
out = {k: round(float(v), 4) for k, v in metrics.items()}
out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "train_acc": round(float(train_acc), 4), "cv_acc": round(float(cv_acc), 4),
            "shuffled_label_cv_acc": round(float(shuf_acc), 4), "overlap_rows": int(overlap),
            "best_single_feature": single.index[0], "best_single_feature_acc": round(float(single.iloc[0]), 4)})
with open("reports/metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nPlots + metrics.json saved to reports/")
