import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

filename = os.path.join("datasets", "urldata.csv")
data0 = pd.read_csv(filename)
data = data0.drop(['Domain'], axis = 1).copy()

fn = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
            'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 
            'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']
cn = ['Good','Bad']

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)

# Separating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)

# Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12)

def _safe_rates(cm: np.ndarray) -> Tuple[float, float, int, int, int, int]:
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return fpr, fnr, tn, fp, fn, tp

def _score_vector(estimator, X) -> Optional[np.ndarray]:
    """Return continuous scores for AUC if possible."""
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        # assume positive class is column 1 in binary classification
        if proba.shape[1] == 2:
            return proba[:, 1]
    return None

def _write_metrics_txt(path: str, name: str, scores: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        f.write(f"Metrics of {name}:\n")
        f.write(f"Accuracy:\t\t{scores['acc']:.3f}\n")
        f.write(f"Precision:\t\t{scores['prec']:.3f}\n")
        f.write(f"Recall:\t\t\t{scores['rec']:.3f}\n")
        f.write(f"F1 Score:\t\t{scores['f1']:.3f}\n")
        f.write(f"False Positive Rate:\t{scores['fpr']:.3f}\n")
        f.write(f"False Negative Rate:\t{scores['fnr']:.3f}\n")
        f.write(f"AUC:\t\t\t{scores['auc']:.3f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f" {scores['tn']:>5} {scores['fp']:>5}\n")
        f.write(f"{scores['fn']:>5}   {scores['tp']:>5}\n")
        f.write("   TN     FP\n   FN     TP\n")
        f.write("\nClassification Report:\n")
        f.write(f"{scores['class_report']}\n")

def _plot_cm(cm: np.ndarray, labels, title: str, out_png: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def _maybe_plot_importance(estimator, X_cols, title_prefix: str) -> None:
    """Plot feature ‘importance’ when the model supports it."""
    # Tree-based feature importances
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        idx = np.argsort(importances)
        plt.figure(figsize=(9, 7))
        plt.barh(range(len(idx)), importances[idx], align="center")
        plt.yticks(range(len(idx)), np.array(X_cols)[idx])
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title(f"{title_prefix} Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_VIZ, f"{title_prefix.lower().replace(' ', '-')}-feature-importance.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # Linear model coefficients (e.g., linear SVM)
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_.ravel()
        plt.figure(figsize=(9, 7))
        plt.barh(range(len(coef)), coef, align="center")
        plt.yticks(np.arange(len(coef)), X_cols)
        plt.xlabel("Coefficient magnitude")
        plt.ylabel("Feature")
        plt.title(f"{title_prefix} Feature Weights")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_VIZ, f"{title_prefix.lower().replace(' ', '-')}-feature-weights.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

def evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Fit, predict, compute & persist metrics, and make standard plots."""
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Standard metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, fnr, tn, fp, fn, tp = _safe_rates(cm)

    # AUC via best available score vector
    scores_vec = _score_vector(estimator, X_test)
    auc = roc_auc_score(y_test, scores_vec) if scores_vec is not None else float("nan")

    # Persist metrics
    metrics_path = os.path.join(OUT_METRICS, f"{name.lower().replace(' ', '')}-metrics.txt")
    _write_metrics_txt(metrics_path, name, {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "fpr": fpr, 
        "fnr": fnr, "auc": auc, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "class_report": classification_report(y_test, y_pred, zero_division=0)
    })

    # Confusion matrix plot
    cm_png = os.path.join(OUT_VIZ, f"{name.lower().replace(' ', '-')}-confusion-matrix.png")
    _plot_cm(cm, getattr(estimator, "classes_", np.unique(y_test)), f"Confusion Matrix - {name}", cm_png)

    # Optional importance/weights
    _maybe_plot_importance(estimator, X_train.columns, name)

    if isinstance(estimator, DecisionTreeClassifier):
        fig, _ = plt.subplots(figsize=(8, 8), dpi=600)
        tree.plot_tree(estimator, feature_names=X_train.columns, class_names=cn, filled=True)
        fig.savefig(os.path.join(OUT_VIZ, "decision-tree.png"), bbox_inches="tight")
        plt.close(fig)

    print(f"{name} Results:")
    print(f"  Accuracy : {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    print(f"  Metrics saved to {metrics_path}")
    print(f"  Visualisations saved to {OUT_VIZ}/")

    return {
        "name": name,
        "estimator": estimator,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "cm": cm,
    }

# ---------- Model registry & execution ----------
models = {}

# Ensure output folders once
OUT_METRICS = "metrics"
OUT_VIZ = "visualisation"
os.makedirs(OUT_METRICS, exist_ok=True)
os.makedirs(OUT_VIZ, exist_ok=True)

if True:  # MLP block toggle
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(50,), solver='lbfgs', alpha=0.01, max_iter=500, random_state=16
    )

if True:  # Linear SVM block toggle
    models["Linear SVM"] = SVC(kernel='linear', C=1, random_state=16)

if True:  # Decision Tree block toggle
    models["Decision Tree"] = DecisionTreeClassifier(max_depth=5, random_state=16)

if True:  # Random Forest block toggle
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300, random_state=16, n_jobs=-1
    )

results = []
for name, clf in models.items():
    evaluate_model(name, clf, X_train, y_train, X_test, y_test)
    # results.append(evaluate_model(name, clf, X_train, y_train, X_test, y_test))