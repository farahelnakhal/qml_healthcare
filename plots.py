from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "quantum_kernel_svm": ("Quantum-kernel SVM", "#6c3fd1", "o", "-"),
    "classical_rbf_svm": ("Classical RBF-SVM (matched)", "#d1483f", "s", "-"),
    "logreg": ("Logistic regression", "#e08a1e", "^", "--"),
    "gbm": ("Gradient boosting", "#2f7d4f", "D", ":"),
}
METRIC = {"roc_auc": "ROC-AUC", "bal_acc": "Balanced accuracy", "f1": "F1"}

def _curve(summary, xs_all, metric, xlabel, title, path, invert_x=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model, (label, color, marker, ls) in STYLE.items():
        if model not in summary or metric not in summary[model]:
            continue
        xs = [x for x in xs_all if x in summary[model][metric]]
        ms = np.array([summary[model][metric][x][0] for x in xs])
        ci = np.array([summary[model][metric][x][1] for x in xs])
        ax.plot(xs, ms, marker=marker, ls=ls, color=color, label=label, lw=1.9, ms=6)
        ax.fill_between(xs, ms - ci, ms + ci, color=color, alpha=0.12)
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC.get(metric, metric))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_feature_scarcity(summary, feature_counts, path, metric="roc_auc"):
    _curve(summary, sorted(feature_counts, reverse=True), metric,
           "Number of features used", "Pima: feature-scarcity robustness (N=150)",
           path, invert_x=False)

def plot_few_shot(summary, train_sizes, path, metric="roc_auc"):
    _curve(summary, sorted(train_sizes), metric,
           "Training set size (N)", "Pima: few-shot learning curve (all 8 features)", path)

def plot_bandwidth(bw, path):
    scales = sorted(bw["auc"])
    auc = np.array([bw["auc"][s][0] for s in scales])
    aucs = np.array([bw["auc"][s][1] for s in scales])
    off = np.array([bw["offdiag"][s][0] for s in scales])
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    c1 = "#6c3fd1"
    ax1.plot(scales, auc, "o-", color=c1, lw=1.9, label="Quantum-kernel ROC-AUC")
    ax1.fill_between(scales, auc - aucs, auc + aucs, color=c1, alpha=0.12)
    ax1.set_xlabel("Feature bandwidth  (angle_scale)")
    ax1.set_ylabel("ROC-AUC", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    c2 = "#888"
    ax2.plot(scales, off, "s--", color=c2, lw=1.6, label="Mean off-diagonal kernel value")
    ax2.set_ylabel("Mean off-diagonal kernel value", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title("Kernel concentration vs bandwidth\n(large bandwidth -> kernel -> identity -> AUC collapses)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
