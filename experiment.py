from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .data import load_raw, make_split, preprocess
from .quantum_kernel import ZZQuantumKernel

C_GRID = [0.1, 1.0, 10.0, 100.0]
BANDWIDTH = 0.05
N_QUBITS_MAX = 8

def _metrics(y, scores, preds):
    return {"roc_auc": roc_auc_score(y, scores),
            "bal_acc": balanced_accuracy_score(y, preds),
            "f1": f1_score(y, preds, zero_division=0)}

def _cv_folds(y_tr):
    return int(min(3, np.bincount(y_tr).min()))

def _tune_C_precomputed(K_tr, y_tr):
    folds = _cv_folds(y_tr)
    if folds < 2:
        return 1.0
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    best_C, best = 1.0, -np.inf
    for C in C_GRID:
        aucs = []
        for tr, va in skf.split(np.zeros(len(y_tr)), y_tr):
            if len(np.unique(y_tr[va])) < 2:
                continue
            clf = SVC(kernel="precomputed", C=C).fit(K_tr[np.ix_(tr, tr)], y_tr[tr])
            aucs.append(roc_auc_score(y_tr[va], clf.decision_function(K_tr[np.ix_(va, tr)])))
        if aucs and np.mean(aucs) > best:
            best, best_C = np.mean(aucs), C
    return best_C

def fit_quantum(X_tr, y_tr, X_test, cols=None, bandwidth=BANDWIDTH):
    nq = len(cols) if cols is not None else N_QUBITS_MAX
    qk = ZZQuantumKernel(n_qubits=nq, reps=1)
    a, b = preprocess(X_tr, X_test, cols, angle_scale=bandwidth)
    K_tr, K_te = qk.gram(a), qk.gram(b, a)
    C = _tune_C_precomputed(K_tr, y_tr)
    clf = SVC(kernel="precomputed", C=C).fit(K_tr, y_tr)
    return clf.decision_function(K_te), clf.predict(K_te)

def fit_classical(kind, X_tr, y_tr, X_test, cols=None):
    a, b = preprocess(X_tr, X_test, cols, angle_scale=None)
    folds = _cv_folds(y_tr)
    if kind == "rbf_svm":
        base = SVC(kernel="rbf", gamma="scale")
        if folds >= 2:
            base = GridSearchCV(base, {"C": C_GRID}, scoring="roc_auc",
                                cv=folds).fit(a, y_tr).best_estimator_
        else:
            base.fit(a, y_tr)
        return base.decision_function(b), base.predict(b)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000).fit(a, y_tr)
        return clf.predict_proba(b)[:, 1], clf.predict(b)
    if kind == "gbm":
        clf = GradientBoostingClassifier(random_state=0).fit(a, y_tr)
        return clf.predict_proba(b)[:, 1], clf.predict(b)
    raise ValueError(kind)

MODELS = {
    "quantum_kernel_svm": lambda Xtr, ytr, Xte, cols: fit_quantum(Xtr, ytr, Xte, cols),
    "classical_rbf_svm": lambda Xtr, ytr, Xte, cols: fit_classical("rbf_svm", Xtr, ytr, Xte, cols),
    "logreg": lambda Xtr, ytr, Xte, cols: fit_classical("logreg", Xtr, ytr, Xte, cols),
    "gbm": lambda Xtr, ytr, Xte, cols: fit_classical("gbm", Xtr, ytr, Xte, cols),
}

def _collect(axis_values, run_one, seeds):
    rows = {}
    for val in axis_values:
        for seed in seeds:
            out = run_one(val, seed)
            for model, m in out.items():
                for k, v in m.items():
                    rows.setdefault(model, {}).setdefault(k, {}).setdefault(val, []).append(v)
    return rows

def experiment_feature_scarcity(feature_counts=(8, 6, 4, 3, 2), N=150, seeds=range(10)):
    X, y = load_raw()

    def run_one(k, seed):
        cols = list(range(k))
        Xtr, ytr, Xte, yte = make_split(X, y, N, seed)
        return {m: _metrics(yte, *fn(Xtr, ytr, Xte, cols)) for m, fn in MODELS.items()}

    return _collect(feature_counts, run_one, list(seeds))

def experiment_few_shot(train_sizes=(10, 16, 24, 40, 80, 150), seeds=range(10)):
    X, y = load_raw()

    def run_one(N, seed):
        Xtr, ytr, Xte, yte = make_split(X, y, N, seed)
        return {m: _metrics(yte, *fn(Xtr, ytr, Xte, None)) for m, fn in MODELS.items()}

    return _collect(train_sizes, run_one, list(seeds))

def experiment_bandwidth(scales=(0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45), N=150, seeds=range(6)):
    X, y = load_raw()
    qk = ZZQuantumKernel(n_qubits=N_QUBITS_MAX, reps=1)
    auc, offdiag = {}, {}
    for s in scales:
        aa, oo = [], []
        for seed in seeds:
            Xtr, ytr, Xte, yte = make_split(X, y, N, seed)
            a, b = preprocess(Xtr, Xte, None, angle_scale=s)
            K_tr, K_te = qk.gram(a), qk.gram(b, a)
            oo.append(K_tr[np.triu_indices_from(K_tr, 1)].mean())
            clf = SVC(kernel="precomputed", C=1.0).fit(K_tr, ytr)
            aa.append(roc_auc_score(yte, clf.decision_function(K_te)))
        auc[s], offdiag[s] = (np.mean(aa), np.std(aa)), (np.mean(oo), np.std(oo))
    return {"auc": auc, "offdiag": offdiag}

def summarize(rows):
    from scipy import stats
    out = {}
    for model, metrics in rows.items():
        out[model] = {}
        for metric, byv in metrics.items():
            out[model][metric] = {}
            for v, vals in byv.items():
                vals = np.asarray(vals, float)
                m = vals.mean()
                ci = (stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
                      if len(vals) > 1 else 0.0)
                out[model][metric][v] = (m, ci)
    return out
