from __future__ import annotations

import os
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "pima-indians-diabetes.data.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
CACHE = os.path.join(DATA_DIR, "pima.csv")

FEATURE_NAMES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                 "insulin", "bmi", "pedigree", "age"]
MISSING_AS_ZERO = [1, 2, 3, 4, 5]

def load_raw():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(DATA_URL, CACHE)
    arr = np.loadtxt(CACHE, delimiter=",")
    X, y = arr[:, :-1].astype(float), arr[:, -1].astype(int)
    for c in MISSING_AS_ZERO:
        X[X[:, c] == 0, c] = np.nan
    return X, y

def make_split(X, y, train_size, seed, test_size=200):
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, _, y_tr, _ = train_test_split(
        X_pool, y_pool, train_size=train_size, stratify=y_pool, random_state=seed)
    return X_tr, y_tr, X_test, y_test

def preprocess(X_tr, X_test, cols=None, angle_scale=None):
    if cols is not None:
        X_tr, X_test = X_tr[:, cols], X_test[:, cols]
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    a = np.where(np.isnan(X_tr), med, X_tr)
    b = np.where(np.isnan(X_test), med, X_test)
    sc = StandardScaler().fit(a)
    a, b = sc.transform(a), sc.transform(b)
    if angle_scale is None:
        return a, b
    hi = angle_scale * np.pi
    mm = MinMaxScaler((0.0, hi)).fit(a)
    return np.clip(mm.transform(a), 0, hi), np.clip(mm.transform(b), 0, hi)
