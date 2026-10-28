from __future__ import annotations

import argparse
import json
import os

from src.experiment import (experiment_bandwidth, experiment_feature_scarcity,
                            experiment_few_shot, summarize)
from src.plots import plot_bandwidth, plot_feature_scarcity, plot_few_shot

RES = os.path.join(os.path.dirname(__file__), "results")
STYLE_ORDER = ["quantum_kernel_svm", "classical_rbf_svm", "logreg", "gbm"]
NAMES = {"quantum_kernel_svm": "Quantum-kernel SVM", "classical_rbf_svm": "RBF-SVM (matched)",
         "logreg": "Logistic regression", "gbm": "Gradient boosting"}

def table(summary, xs, metric, axis_label):
    lines = [f"\n=== {metric}: mean +/- 95% CI  ({axis_label}) ==="]
    lines.append("model".ljust(22) + "".join(f"{axis_label}={x}".ljust(16) for x in xs))
    for m in STYLE_ORDER:
        if m not in summary or metric not in summary[m]:
            continue
        row = NAMES[m].ljust(22)
        for x in xs:
            if x in summary[m][metric]:
                mu, ci = summary[m][metric][x]
                row += f"{mu:.3f}±{ci:.3f}".ljust(16)
            else:
                row += "".ljust(16)
        lines.append(row)
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    seeds = range(3) if args.quick else range(10)
    fcs = (8, 4, 2) if args.quick else (8, 6, 4, 3, 2)
    Ns = (10, 40, 150) if args.quick else (10, 16, 24, 40, 80, 150)

    print("Running feature-scarcity experiment ...")
    scar = summarize(experiment_feature_scarcity(fcs, seeds=seeds))
    print("Running few-shot experiment ...")
    shot = summarize(experiment_few_shot(Ns, seeds=seeds))
    print("Running bandwidth analysis ...")
    bw = experiment_bandwidth(seeds=range(3) if args.quick else range(6))

    plot_feature_scarcity(scar, fcs, os.path.join(RES, "feature_scarcity_auc.png"))
    plot_few_shot(shot, Ns, os.path.join(RES, "few_shot_auc.png"))
    plot_bandwidth(bw, os.path.join(RES, "bandwidth_concentration.png"))

    report = []
    report.append(table(scar, sorted(fcs, reverse=True), "roc_auc", "k"))
    report.append(table(scar, sorted(fcs, reverse=True), "bal_acc", "k"))
    report.append(table(shot, sorted(Ns), "roc_auc", "N"))
    text = "\n".join(report)
    print(text)
    with open(os.path.join(RES, "tables.txt"), "w") as f:
        f.write(text + "\n")

    serial = {"feature_scarcity": _ser(scar), "few_shot": _ser(shot),
              "bandwidth": {"auc": {str(k): list(v) for k, v in bw["auc"].items()},
                            "offdiag": {str(k): list(v) for k, v in bw["offdiag"].items()}}}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\nWrote plots, tables.txt and summary.json to {RES}/")

def _ser(summary):
    return {m: {met: {str(x): list(v) for x, v in byx.items()}
                for met, byx in mets.items()} for m, mets in summary.items()}

if __name__ == "__main__":
    main()
