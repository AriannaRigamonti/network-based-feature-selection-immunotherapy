#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Script: 2_cross_study_prediction.py
# Author: Arianna Rigamonti
# Description: Cross-study immunotherapy response prediction. Trains on one
#              cohort and evaluates on external cohorts across multiple feature
#              modes (NetBio, Wallenius, GSEA, Forward FS on all ssGSEA, and
#              single-gene PD1/PD-L1/CTLA4) and models (LR, RF, SVC). Exports
#              summary metrics with confidence intervals, per-sample predictions,
#              per-sample SHAP values, and PCA panels (pre/post z-score).
#
# Outputs under: results/2_cross_study_prediction/{VERSION}_{INTERACTION}/
#   - summary_metrics_<train>.tsv
#   - predictions_<train>.tsv
#   - shap/<train>_to_<test>/<FeatureSet>/<Model>/SHAP_<Sample>.tsv
#   - pca/{genes|pathways}_{train}_{test}_{pre|post}.png
#   - selected_features/<train>_to_<test>_FWD_FS_features.txt
################################################################################

"""
Cross-study immunotherapy response prediction.

Supports:
- Feature modes: NetBio, NetBio_Wallenius, GSEA, FS_AIC(AllSSGSEA),
  plus single-gene features PD1, PD-L1, CTLA4.
- Models: LogisticRegression, RandomForest, SVC.
- Saves metrics, per-sample predictions, SHAP values.

"""

# =============================================================================
# Imports & warnings
# =============================================================================
import os
import sys
import time
import glob
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Dict
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm

from sklearn.exceptions import UndefinedMetricWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import HessianInversionWarning

# ---- Warning policy (silence expected non-critical runtime warnings) ----
warnings.filterwarnings("ignore", category=HessianInversionWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


# =============================================================================
# Global configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)
N_JOBS = 10
DATA_DIR = "data/immuno"

# Forward Selection (AIC) defaults
FS_DELTA  = 0.01
FS_NJOBS  = 10

# Cohort registry and target mapping (for feature file resolution)
STUDIES = ["Gide", "Liu", "Huang", "Kim", "IMvigor210",
           "Auslander", "Riaz", "Prat", "PEOPLE", "Ravi"]

TARGET_BY_STUDY = {
    "Gide": "PD1_CTLA4",
    "Liu": "PD1",
    "Huang": "PD1",
    "Kim": "PD1",
    "IMvigor210": "PD-L1",
    "Auslander": "PD1_CTLA4",
    "Riaz": "PD1",
    "Prat": "PD1",
    "PEOPLE": "PD1",
    "Ravi": "PD1_PD-L1_CTLA4",
}

# Single-gene aliases to HGNC
GENE_ALIASES = {
    "PD1": "PDCD1",
    "PD-L1": "CD274",
    "CTLA4": "CTLA4"
}


# =============================================================================
# Helpers: time + logging
# =============================================================================
def ts():
    """Timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fmt_secs(sec):
    """Pretty print seconds as H:MM:SS."""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def log(msg, *, end="\n", file=sys.stdout):
    """Console log with timestamp."""
    print(f"[{ts()}] {msg}", end=end, file=file)


# =============================================================================
# Confidence intervals and bootstrap utilities
# =============================================================================
def wilson_ci(successes: int, total: int, alpha: float = 0.05):
    """Wilson score interval for binomial proportions (accuracy, precision, etc.)."""
    if total == 0:
        return (np.nan, np.nan)
    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return float(low), float(high)

def bootstrap_ci_metric(y_true, y_pred, scorer, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for metrics computed on (y_true, y_pred)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        vals.append(scorer(y_true[s], y_pred[s]))
    vals = np.array(vals, dtype=float)
    lo = np.percentile(vals, 100*alpha/2)
    hi = np.percentile(vals, 100*(1-alpha/2))
    return float(lo), float(hi)

def bootstrap_ci_auc(y_true, y_score, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for ROC-AUC based on predicted probabilities."""
    def _safe_auc(y, p):
        try:
            return roc_auc_score(y, p)
        except Exception:
            return np.nan
    if rng is None:
        rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        auc = _safe_auc(y_true[s], y_score[s])
        if np.isfinite(auc):
            vals.append(auc)
    if len(vals) == 0:
        return (np.nan, np.nan)
    vals = np.array(vals, dtype=float)
    lo = np.percentile(vals, 100*alpha/2)
    hi = np.percentile(vals, 100*(1-alpha/2))
    return float(lo), float(hi)


# =============================================================================
# Data loading
# =============================================================================
def _expr_path_for_study(study: str) -> str:
    """Return expression file path (Kim uses normalized file)."""
    base = f"{DATA_DIR}/{study}"
    return os.path.join(base, f"{study}_mRNA.norm3.txt") if study == "Kim" else os.path.join(base, f"{study}_mRNA.txt")

def load_gene_expression(study: str) -> pd.DataFrame:
    """Load gene expression matrix (genes x samples); genes upper-cased."""
    df = pd.read_csv(_expr_path_for_study(study), sep="\t", index_col=0)
    df.index = df.index.astype(str).str.upper()
    return df

def load_single_gene_feature(study: str, gene: str) -> pd.DataFrame:
    """Load single gene as (samples x 1) feature matrix."""
    ge = load_gene_expression(study)
    gene = GENE_ALIASES.get(gene.upper(), gene.upper())
    if gene not in ge.index:
        raise ValueError(f"{gene} not found in {study} expression matrix")
    x = ge.loc[[gene]].T
    x.columns = [gene]
    return x, [gene]

def load_ssgsea_matrix(study: str) -> pd.DataFrame:
    """Load ssGSEA NES (pathways x samples) unfiltered matrix."""
    p = os.path.join(DATA_DIR, study, f"{study}_ssGSEA.txt")
    df = pd.read_csv(p, sep="\t", index_col=0)
    return df

def load_labels(study: str) -> pd.Series:
    """Load binary response labels; asserts 0/1 encoding."""
    data_dir = DATA_DIR
    fldr = study
    for tmp in os.listdir(DATA_DIR):
        if (study in tmp) and (not tmp.endswith(".txt")):
            fldr = tmp
            break
    p = os.path.join(data_dir, fldr, f"{study}_metadata.txt")
    df = pd.read_csv(p, sep="\t")
    s = pd.Series(
        df["Response"].values,
        index=df["Patient"].astype(str).values,
        name="Response"
    )
    vals = pd.unique(s)
    if not set(map(str, vals)).issubset({"0", "1"}):
        raise ValueError(f"{study}: Response should contain only 0/1, found {list(vals)}.")
    
    print(f"[CHECK] {study} metadata: {s.shape[0]} samples, "
          f"{(s.astype(int)==1).sum()} responders, {(s.astype(int)==0).sum()} non-responders")

    return s

def load_gsea_features(study: str, target: str, version: str, interaction: str):
    """Load GSEA preranked features ({study}_ssGSEA_{target}_{version}_{interaction}.txt)."""
    p = os.path.join(DATA_DIR, study, f"{study}_ssGSEA_{target}_{version}_{interaction}.txt")
    nes = pd.read_csv(p, sep="\t", index_col=0)
    X = nes.T.apply(pd.to_numeric, errors="coerce")
    return X, list(X.columns)

def load_netbio_features(study: str, target: str, fo_base: str):
    """Load NetBio (hypergeometric ORA) ssGSEA features from preprocessing results."""
    netbio_root = os.path.join(fo_base, "NetBio_original")
    pattern = os.path.join(netbio_root, study, f"{study}_ssGSEA_{target}_NetBio_features.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NetBio data for {study}. Expected: {pattern}")
    df = pd.read_csv(files[0], sep="\t", index_col=0)
    X = df.T.apply(pd.to_numeric, errors="coerce")
    return X, list(X.columns)

def load_wallenius_features(study: str, target: str, fo_base: str):
    """Load NetBio_Wallenius (weighted ORA) ssGSEA features from preprocessing results."""
    w_root = os.path.join(fo_base, "NetBio_wallenius")
    pattern = os.path.join(w_root, study, f"{study}_ssGSEA_{target}_Wallenius_features.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Wallenius data for {study}. Expected: {pattern}")
    df = pd.read_csv(files[0], sep="\t", index_col=0)
    X = df.T.apply(pd.to_numeric, errors="coerce")
    return X, list(X.columns)


# =============================================================================
# Models (LR, SVC, RF) and grids
# =============================================================================
def make_lr_pipeline():
    """Logistic Regression with z-score scaling and L2 penalty."""
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=10000,
            class_weight="balanced", random_state=SEED
        ))
    ])
    grid = {"clf__C": np.round(np.concatenate([np.linspace(0.01, 0.1, 10),
                                               np.linspace(0.2, 1.5, 14)]), 3)}
    return pipe, grid

def make_svc_pipeline():
    """SVC with probability=True, supporting linear and RBF kernels."""
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SVC(probability=True, class_weight="balanced", random_state=SEED))
    ])
    grid = {
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.01, 0.1, 1, 5, 10, 50, 100],
        "clf__gamma": ["scale"],
    }
    return pipe, grid

def make_rf_pipeline():
    """Random Forest with balanced class weights and many trees."""
    pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=1000, class_weight="balanced",
            random_state=SEED, n_jobs=N_JOBS
        ))
    ])
    grid = {
        "clf__max_depth": [None, 3, 5, 7, 9],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    }
    return pipe, grid

MODEL_FACTORIES = {
    "LogisticRegression": make_lr_pipeline,
    "SVC": make_svc_pipeline,
    "RandomForest": make_rf_pipeline,
}

# ----------------------------- SHAP helper -----------------------------
from sklearn.pipeline import Pipeline  # (kept as in original script)

def compute_shap_single(model_name, best_estimator, X_tr_sel, X_te_sel, feat_names, random_state=SEED):
    """
    Compute SHAP values for one test sample (cross-study).
    Returns dict with 'features', 'shap_values', and 'base_value'.
    Handles Pipelines with scalers and adapts to different SHAP output shapes.
    """
    import shap
    from shap import maskers, links
    import numpy as np

    def _transform_for_model(estimator, X):
        if isinstance(estimator, Pipeline) and 'scaler' in estimator.named_steps:
            return estimator.named_steps['scaler'].transform(X)
        return X

    def _as_float(x):
        return np.asarray(x, dtype=np.float64)

    def _pick_shap_vector(shap_vals, n_features):
        # SHAP can return lists (per class) or arrays with various shapes
        if isinstance(shap_vals, list):
            arr = np.asarray(shap_vals[-1], dtype=float)  # positive class
        else:
            arr = np.asarray(shap_vals, dtype=float)

        if arr.ndim == 1 and arr.size == n_features:
            return arr
        if arr.ndim == 2:
            if arr.shape[0] == 1 and arr.shape[1] == n_features:
                return arr[0]
            if arr.shape[1] == n_features:  # (classes, features)
                cls_idx = 1 if arr.shape[0] >= 2 else 0
                return arr[cls_idx]
        if arr.ndim == 3 and arr.shape[0] == 1:
            if arr.shape[2] == n_features:
                cls_idx = 1 if arr.shape[1] >= 2 else 0
                return arr[0, cls_idx, :]
            if arr.shape[1] == n_features:
                cls_idx = 1 if arr.shape[2] >= 2 else 0
                return arr[0, :, cls_idx]
        for axis in range(arr.ndim):
            if arr.shape[axis] == n_features:
                index = [0] * arr.ndim
                index[axis] = slice(None)
                vec = arr[tuple(index)]
                return np.asarray(vec, dtype=float).ravel()
        raise ValueError(f"Unexpected SHAP shape {arr.shape} vs n_features={n_features}")

    # Prepare matrices
    X_train_m = _as_float(_transform_for_model(best_estimator, X_tr_sel))
    X_test_m  = _as_float(_transform_for_model(best_estimator, X_te_sel))
    n_features = X_train_m.shape[1]

    clf = best_estimator
    if isinstance(best_estimator, Pipeline):
        clf = best_estimator.named_steps['clf']

    if model_name == "RandomForest":
        explainer = shap.TreeExplainer(clf, data=X_train_m)
        shap_vals = explainer.shap_values(X_test_m)
        sv = _pick_shap_vector(shap_vals, n_features)
        ev = np.array(explainer.expected_value, dtype=float)
        base_val = float(ev[1] if ev.ndim > 0 and ev.size >= 2 else ev.ravel()[0])

    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(
            clf, maskers.Independent(X_train_m), link=links.logit
        )
        shap_vals = explainer.shap_values(X_test_m)
        sv = _pick_shap_vector(shap_vals, n_features)
        base_val = float(np.array(explainer.expected_value).ravel()[0])

    elif model_name == "SVC":
        if getattr(clf, "kernel", None) == "linear":
            explainer = shap.LinearExplainer(
                clf, maskers.Independent(X_train_m), link=links.identity
            )
            shap_vals = explainer.shap_values(X_test_m)
            sv = _pick_shap_vector(shap_vals, n_features)
            base_val = float(np.array(explainer.expected_value).ravel()[0])
        else:
            from sklearn.cluster import KMeans
            X_tr_raw = _as_float(X_tr_sel)
            X_te_raw = _as_float(X_te_sel)
            n_bg = min(50, X_tr_raw.shape[0])
            bg = KMeans(n_clusters=n_bg, random_state=random_state).fit(X_tr_raw).cluster_centers_ \
                 if X_tr_raw.shape[0] > n_bg else X_tr_raw
            def f_prob(x):
                return best_estimator.predict_proba(x)[:, 1]
            explainer = shap.KernelExplainer(f_prob, bg)
            shap_vals = explainer.shap_values(X_te_raw, nsamples="auto")
            sv = _pick_shap_vector(shap_vals, n_features)
            base_val = float(np.array(explainer.expected_value).ravel()[0])

    else:
        raise ValueError(f"Unsupported model for SHAP: {model_name}")

    return {
        "features": list(feat_names),
        "shap_values": np.asarray(sv, dtype=float).ravel().tolist(),
        "base_value": float(base_val),
    }


# =============================================================================
# Alignment utilities
# =============================================================================
def align_xy(X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series):
    """Align samples and intersect features across train/test."""
    common_tr = X_train.index.intersection(y_train.index)
    common_te = X_test.index.intersection(y_test.index)
    X_train = X_train.loc[common_tr].copy()
    y_train = y_train.loc[common_tr].astype(int).copy()
    X_test  = X_test.loc[common_te].copy()
    y_test  = y_test.loc[common_te].astype(int).copy()
    common_feats = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_feats].copy()
    X_test  = X_test[common_feats].copy()
    return X_train, y_train, X_test, y_test, list(common_feats)


# =============================================================================
# Forward Selection (AIC) helpers
# =============================================================================
def _fit_logit_and_aic(X_sm, y) -> float:
    """Fit logistic regression (statsmodels) and return AIC; inf on failure."""
    try:
        res = sm.Logit(y, X_sm).fit(method="lbfgs", disp=False, maxiter=500)
        return float(res.aic) if res.aic is not None else np.inf
    except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError, ValueError):
        return np.inf

def _forward_step_aic(X, y, selected, feat) -> Tuple[int, float]:
    """Evaluate AIC for adding one candidate feature to the current subset."""
    idx = selected + [feat]
    aic_val = _fit_logit_and_aic(sm.add_constant(X[:, idx]), y)
    return feat, aic_val

def forward_sfs_aic(X: np.ndarray, y: np.ndarray,
                    delta: float = FS_DELTA, n_jobs: int = 1,
                    return_trace: bool = False):
    """Greedy forward selection with AIC stopping rule."""
    selected, remaining = [], list(range(X.shape[1]))
    last_aic = np.inf
    trace: List[Tuple[int, float]] = []
    while remaining:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_forward_step_aic)(X, y, selected, f) for f in remaining
        )
        best_f, best_aic = min(results, key=lambda x: x[1])
        selected.append(best_f); remaining.remove(best_f)
        trace.append((len(selected), best_aic))
        if last_aic - best_aic < delta:
            break
        last_aic = best_aic
    return (selected, trace) if return_trace else selected


# =============================================================================
# Main train/test routine (single train→test, one feature mode)
# =============================================================================
def run_train_test_once(train_study, test_study, feature_mode, version, interaction,
                        models, out_roots, debug=True):
    """
    Train on 'train_study' and evaluate on 'test_study' for a given feature mode.
    Exports SHAP per-sample files; returns aggregated metrics and per-sample predictions.
    """
    log(f"Running {train_study}->{test_study} | features={feature_mode}")

    y_train = load_labels(train_study)
    y_test  = load_labels(test_study)

    log(f"{train_study} labels: {y_train.shape[0]} samples, "
    f"{y_train.sum()} responders / {(y_train==0).sum()} non-responders")
    log(f"{test_study} labels: {y_test.shape[0]} samples, "
    f"{y_test.sum()} responders / {(y_test==0).sum()} non-responders")

    if len(np.unique(y_train)) < 2:
        log(f"[ERROR] {train_study}: y_train has only one class: {np.unique(y_train)}")
    if len(np.unique(y_test)) < 2:
        log(f"[ERROR] {test_study}: y_test has only one class: {np.unique(y_test)}")

    # ---- Feature construction per mode ----
    if feature_mode in ["PDCD1", "CD274", "CTLA4"]:
        Xtr, _ = load_single_gene_feature(train_study, feature_mode)
        Xte, _ = load_single_gene_feature(test_study, feature_mode)
        Xtr, ytr, Xte, yte, feats = align_xy(Xtr, y_train, Xte, y_test)
        if Xtr.shape[1] == 0:
            log(f"[SKIP] {train_study}->{test_study} [{feature_mode}]: no overlapping features")
            return [], []

        log(f"Aligned: {Xtr.shape[0]} train, {Xte.shape[0]} test samples, "f"{Xtr.shape[1]} features")

    elif feature_mode == "NetBio":
        Xtr, _ = load_netbio_features(train_study, TARGET_BY_STUDY[train_study], out_roots["FO_BASE"])
        Xte, _ = load_netbio_features(test_study, TARGET_BY_STUDY[test_study], out_roots["FO_BASE"])
        Xtr, ytr, Xte, yte, feats = align_xy(Xtr, y_train, Xte, y_test)
        log(f"Aligned: {Xtr.shape[0]} train, {Xte.shape[0]} test samples, "f"{Xtr.shape[1]} features")

    elif feature_mode == "NetBio_Wallenius":
        Xtr, _ = load_wallenius_features(train_study, TARGET_BY_STUDY[train_study], out_roots["FO_BASE"])
        Xte, _ = load_wallenius_features(test_study, TARGET_BY_STUDY[test_study], out_roots["FO_BASE"])
        Xtr, ytr, Xte, yte, feats = align_xy(Xtr, y_train, Xte, y_test)
        log(f"Aligned: {Xtr.shape[0]} train, {Xte.shape[0]} test samples, "f"{Xtr.shape[1]} features")

    elif feature_mode == "GSEA":
        Xtr, _ = load_gsea_features(train_study, TARGET_BY_STUDY[train_study], version, interaction)
        Xte, _ = load_gsea_features(test_study, TARGET_BY_STUDY[test_study], version, interaction)
        Xtr, ytr, Xte, yte, feats = align_xy(Xtr, y_train, Xte, y_test)
        log(f"Aligned: {Xtr.shape[0]} train, {Xte.shape[0]} test samples, "f"{Xtr.shape[1]} features")
    
    elif feature_mode == "FWD_FS":
        # Load full ssGSEA NES (pathways x samples), transpose to (samples x pathways)
        nes_tr = load_ssgSEA_matrix := load_ssgsea_matrix  # alias for readability
        nes_tr = load_ssgSEA_matrix(train_study).T
        nes_te = load_ssgSEA_matrix(test_study).T

        # Keep only common pathways
        common_feats = nes_tr.columns.intersection(nes_te.columns)
        nes_tr = nes_tr[common_feats].copy()
        nes_te = nes_te[common_feats].copy()

        # Align samples to labels
        common_tr = nes_tr.index.intersection(y_train.index)
        common_te = nes_te.index.intersection(y_test.index)
        Xtr_full = nes_tr.loc[common_tr].copy()
        ytr = y_train.loc[common_tr].astype(int).copy()
        Xte_full = nes_te.loc[common_te].copy()
        yte = y_test.loc[common_te].astype(int).copy()

        # Z-score training data for FS step only
        scaler_fs = StandardScaler(with_mean=True, with_std=True).fit(Xtr_full.values)
        Xtr_z = scaler_fs.transform(Xtr_full.values)

        # Forward selection on training set (restricted to common pathways)
        sel_idx, trace = forward_sfs_aic(
            Xtr_z, ytr.values, delta=FS_DELTA, n_jobs=FS_NJOBS, return_trace=True
        )
        if len(sel_idx) == 0:
            log(f"[SKIP] {train_study}->{test_study} [FWD_FS]: no features selected")
            return [], []

        sel_feats = [Xtr_full.columns[i] for i in sel_idx]

        # Restrict to selected features
        Xtr = Xtr_full[sel_feats].copy()
        Xte = Xte_full[sel_feats].copy()
        feats = sel_feats

        log(f"Aligned: {Xtr.shape[0]} train, {Xte.shape[0]} test samples, {Xtr.shape[1]} FWD_FS features")

        # Save selected features
        out_dir_fs = os.path.join("results/2_cross_study_prediction",
                                f"{version}_{interaction}", "selected_features")
        os.makedirs(out_dir_fs, exist_ok=True)
        out_file = os.path.join(out_dir_fs, f"{train_study}_to_{test_study}_FWD_FS_features.txt")
        with open(out_file, "w") as f:
            for feat in sel_feats:
                f.write(feat + "\n")
        log(f"Saved selected features to {out_file}")

    # ---- Train & evaluate for each requested model ----
    rows, preds = [], []
    for mdl in models:
        pipe, grid = MODEL_FACTORIES[mdl]()
        gcv = GridSearchCV(estimator=pipe, param_grid=grid, scoring="roc_auc",
                           cv=5, n_jobs=N_JOBS, refit=True, verbose=0)
        gcv.fit(Xtr.values, ytr.values)

        log(f"[{mdl}] best params: {gcv.best_params_}, "
        f"cv AUC={gcv.best_score_:.3f}")

        y_pred = gcv.best_estimator_.predict(Xte.values)

        if len(np.unique(y_pred)) < 2:
            log(f"[{mdl}] WARNING: model predicted only one class ({np.unique(y_pred)[0]}) "
                f"on {test_study} ({len(y_pred)} samples)")
        
        try:
            y_prob = gcv.best_estimator_.predict_proba(Xte.values)[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)

        # ---- SHAP computation and saving (per-sample) ----
        try:
            shap_dir = os.path.join(
                "results/2_cross_study_prediction",
                f"{version}_{interaction}",
                "shap",
                f"{train_study}_to_{test_study}",
                str(feature_mode),
                mdl
            )
            os.makedirs(shap_dir, exist_ok=True)

            feat_names = list(feats)

            for i, samp in enumerate(Xte.index.tolist()):
                X_tr_sel = Xtr.values
                X_te_sel = Xte.values[[i], :]  # single sample

                shap_out = compute_shap_single(
                    model_name=mdl,
                    best_estimator=gcv.best_estimator_,
                    X_tr_sel=X_tr_sel,
                    X_te_sel=X_te_sel,
                    feat_names=feat_names,
                    random_state=SEED
                )

                shap_df = pd.DataFrame({
                    "Feature": shap_out["features"],
                    "SHAP": shap_out["shap_values"]
                })
                shap_df["Sample"] = samp
                shap_path = os.path.join(shap_dir, f"SHAP_{samp}.tsv")
                shap_df.to_csv(shap_path, sep="\t", index=False)

            log(f"[{mdl}] SHAP saved to {shap_dir}")

        except Exception as e:
            log(f"[{mdl}] SHAP WARN {train_study}->{test_study} [{feature_mode}]: {e}")
        # --------------------------------------------------

        # ---- Metrics & CIs ----
        tn, fp, fn, tp = confusion_matrix(yte.values, y_pred, labels=[0, 1]).ravel()
        auc = roc_auc_score(yte.values, y_prob) if len(np.unique(yte)) > 1 else np.nan
        ap = average_precision_score(yte.values, y_prob) if len(np.unique(yte)) > 1 else np.nan
        acc = accuracy_score(yte.values, y_pred)
        bal_acc = balanced_accuracy_score(yte.values, y_pred)
        prec = precision_score(yte.values, y_pred, zero_division=0)
        rec = recall_score(yte.values, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1_bin = f1_score(yte.values, y_pred, zero_division=0)
        f1_macro = f1_score(yte.values, y_pred, average="macro", zero_division=0)

        log(f"[{mdl}] test AUC={auc:.3f}, ACC={acc:.3f}, "
        f"Bal_ACC={bal_acc:.3f}, F1={f1_bin:.3f}")
        
        # Confidence intervals
        acc_ci = wilson_ci(tp + tn, len(yte))
        prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (np.nan, np.nan)
        rec_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (np.nan, np.nan)
        spec_ci = wilson_ci(tn, tn + fp) if (tn + fp) > 0 else (np.nan, np.nan)
        f1_bin_ci = bootstrap_ci_metric(yte.values, y_pred,
                                        lambda yt, yp: f1_score(yt, yp, zero_division=0))
        f1_macro_ci = bootstrap_ci_metric(yte.values, y_pred,
                                          lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
        bal_acc_ci = bootstrap_ci_metric(yte.values, y_pred,
                                         lambda yt, yp: balanced_accuracy_score(yt, yp))
        auc_ci = bootstrap_ci_auc(yte.values, y_prob) if np.isfinite(auc) else (np.nan, np.nan)
        ap_ci = bootstrap_ci_metric(yte.values, y_prob,
                                    lambda yt, p: average_precision_score(yt, p)) if np.isfinite(ap) else (np.nan, np.nan)

        # Summary row
        rows.append({
            "Train": train_study,
            "Test": test_study,
            "Features": feature_mode,
            "Model": mdl,
            "AUC": auc,
            "AUC_CI_low": auc_ci[0], "AUC_CI_high": auc_ci[1],
            "PR_AUC": ap,
            "PR_AUC_CI_low": ap_ci[0], "PR_AUC_CI_high": ap_ci[1],
            "ACC": acc,
            "ACC_CI_low": acc_ci[0], "ACC_CI_high": acc_ci[1],
            "Balanced_ACC": bal_acc,
            "Balanced_ACC_CI_low": bal_acc_ci[0], "Balanced_ACC_CI_high": bal_acc_ci[1],
            "Precision": prec,
            "Prec_CI_low": prec_ci[0], "Prec_CI_high": prec_ci[1],
            "Recall": rec,
            "Recall_CI_low": rec_ci[0], "Recall_CI_high": rec_ci[1],
            "Specificity": spec,
            "Spec_CI_low": spec_ci[0], "Spec_CI_high": spec_ci[1],
            "F1": f1_bin,
            "F1_CI_low": f1_bin_ci[0], "F1_CI_high": f1_bin_ci[1],
            "F1_macro": f1_macro,
            "F1_macro_CI_low": f1_macro_ci[0], "F1_macro_CI_high": f1_macro_ci[1],
        })

        # Per-sample predictions
        for s, yt, yp, pr in zip(Xte.index.tolist(), yte.values.tolist(),
                                 y_pred.tolist(), y_prob.tolist()):
            preds.append({
                "Train": train_study,
                "Test": test_study,
                "Features": feature_mode,
                "Model": mdl,
                "Sample": s,
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_prob": float(pr)
            })
    return rows, preds


# =============================================================================
# PCA helpers (visual QC of distribution shift)
# =============================================================================
def _pca_scatter(train_X: np.ndarray, test_X: np.ndarray,
                 train_ids: List[str], test_ids: List[str],
                 out_path: str, title: str, n_components: int = 2):
    """
    Compute PCA (on the concatenated matrix) and save a 2D scatter plot with cohort labels.
    """
    X = np.vstack([train_X, test_X])
    pca = PCA(n_components=n_components, random_state=SEED)
    Z = pca.fit_transform(X)
    z_tr = Z[:train_X.shape[0], :]
    z_te = Z[train_X.shape[0]:, :]

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(z_tr[:, 0], z_tr[:, 1], s=40, alpha=0.85,
                label="Train", edgecolor="none")
    plt.scatter(z_te[:, 0], z_te[:, 1], s=40, alpha=0.85,
                label="Test", marker="^", edgecolor="none")

    var = pca.explained_variance_ratio_

    # Labels and title
    plt.xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=18, labelpad=10)
    plt.ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=18, labelpad=10)
    plt.title(title, fontsize=18, pad=12)

    # Ticks / legend
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(frameon=False, loc="best", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def _align_samples_and_features_for_pca(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Align features (rows) and return matrices as (n_samples, n_features) for PCA.
    """
    common_features = train_df.index.intersection(test_df.index)
    train_df = train_df.loc[common_features]
    test_df  = test_df.loc[common_features]
    Xtr = train_df.T.values
    Xte = test_df.T.values
    tr_ids = list(train_df.columns)
    te_ids = list(test_df.columns)
    return Xtr, tr_ids, Xte, te_ids, list(common_features)

def _zscore_within_cohort(X: np.ndarray):
    """Feature-wise z-score within each cohort (mean/std over samples)."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(X)

def generate_pca_panels(train_study: str, test_study: str,
                        version: str, interaction: str,
                        out_root: str):
    """
    Create four PCA plots for a given train→test pair:
      - genes_pre, genes_post ; pathways_pre, pathways_post
    """
    out_dir = os.path.join(out_root, f"{version}_{interaction}", "pca")
    os.makedirs(out_dir, exist_ok=True)

    # Genes (mRNA)
    ge_tr = load_gene_expression(train_study)
    ge_te = load_gene_expression(test_study)
    Xtr_g, tr_ids_g, Xte_g, te_ids_g, _ = _align_samples_and_features_for_pca(ge_tr, ge_te)

    _pca_scatter(
        Xtr_g, Xte_g, tr_ids_g, te_ids_g,
        out_path=os.path.join(out_dir, f"genes_{train_study}_{test_study}_pre.png"),
        title=f"Genes PCA (pre) | {train_study}→{test_study}"
    )

    Xtr_g_post = _zscore_within_cohort(Xtr_g)
    Xte_g_post = _zscore_within_cohort(Xte_g)
    _pca_scatter(
        Xtr_g_post, Xte_g_post, tr_ids_g, te_ids_g,
        out_path=os.path.join(out_dir, f"genes_{train_study}_{test_study}_post.png"),
        title=f"Genes PCA (post) | {train_study}→{test_study}"
    )

    # Pathways (ssGSEA)
    pw_tr = load_ssgsea_matrix(train_study)
    pw_te = load_ssgsea_matrix(test_study)
    Xtr_p, tr_ids_p, Xte_p, te_ids_p, _ = _align_samples_and_features_for_pca(pw_tr, pw_te)

    _pca_scatter(
        Xtr_p, Xte_p, tr_ids_p, te_ids_p,
        out_path=os.path.join(out_dir, f"pathways_{train_study}_{test_study}_pre.png"),
        title=f"Pathways PCA (pre) | {train_study}→{test_study}"
    )

    Xtr_p_post = _zscore_within_cohort(Xtr_p)
    Xte_p_post = _zscore_within_cohort(Xte_p)
    _pca_scatter(
        Xtr_p_post, Xte_p_post, tr_ids_p, te_ids_p,
        out_path=os.path.join(out_dir, f"pathways_{train_study}_{test_study}_post.png"),
        title=f"Pathways PCA (post) | {train_study}→{test_study}"
    )


# =============================================================================
# Cross-study runner (multi test cohorts, multi feature modes)
# =============================================================================
def run_cross_study(
    version: str = "v11",
    interaction: str = "full",
    train_study: str = "Gide",
    test_studies: List[str] = None,
    models: List[str] = None,
    feature_modes: List[str] = None,
    out_root: str = "results/2_cross_study_prediction",
    debug: bool = True
):
    """
    Orchestrate a full cross-study benchmark: optional PCA panels,
    then train→test across feature modes and models; return DataFrames.
    """
    if test_studies is None:
        test_studies = ["Auslander", "Prat", "Riaz"]
    if models is None:
        models = ["LogisticRegression", "RandomForest", "SVC"]
    if feature_modes is None:
        feature_modes = ["NetBio", "NetBio_Wallenius", "GSEA", "FWD_FS",
                         "PDCD1", "CD274", "CTLA4"]
    
    log(f"=== RUN: train={train_study}, tests={test_studies}, "
    f"features={feature_modes}, models={models} ===")

    fo_base = f"results/0_data_collection_and_preprocessing/{version}/{interaction}"
    out_roots = dict(FO_BASE=fo_base)

    all_rows, all_preds = [], []
    
    for test_study in test_studies:
        try:
            generate_pca_panels(train_study, test_study,
                                version, interaction, out_root)
        except Exception as e:
            log(f"[WARN] PCA {train_study}->{test_study} failed: {e}")

        for feat_mode in feature_modes:
            try:
                rows, preds = run_train_test_once(
                    train_study, test_study, feat_mode,
                    version, interaction, models, out_roots, debug
                )
                all_rows.extend(rows)
                all_preds.extend(preds)
            except Exception as e:
                log(f"[FAIL] {train_study}->{test_study} [{feat_mode}]: {e}")

    res_df = pd.DataFrame(all_rows)
    pred_df = pd.DataFrame(all_preds)
    return res_df, pred_df


# =============================================================================
# CLI entry point
# =============================================================================
def main():
    """CLI wrapper to launch a cross-study run and persist outputs."""
    parser = argparse.ArgumentParser(description="Cross-study immunotherapy prediction")
    parser.add_argument("--train", type=str, required=True, help="Training study")
    parser.add_argument("--tests", type=str, nargs="+", required=True, help="Test studies")
    parser.add_argument("--version", type=str)
    parser.add_argument("--interaction", type=str)
    parser.add_argument("--models", type=str, nargs="+",
                        default=["LogisticRegression", "RandomForest", "SVC"])
    parser.add_argument("--features", type=str, nargs="+",
                        default=["NetBio", "NetBio_Wallenius", "GSEA", "FWD_FS",
                                 "PDCD1", "CD274", "CTLA4"])
    parser.add_argument("--out", type=str, default="results/2_cross_study_prediction")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    res_df, pred_df = run_cross_study(
        version=args.version,
        interaction=args.interaction,
        train_study=args.train,
        test_studies=args.tests,
        models=args.models,
        feature_modes=args.features,
        out_root=args.out,
        debug=args.debug
    )

    # Save outputs
    os.makedirs(args.out, exist_ok=True)

    res_path = os.path.join(args.out, f"{args.version}_{args.interaction}/summary_metrics_{args.train}.tsv")
    pred_path = os.path.join(args.out, f"{args.version}_{args.interaction}/predictions_{args.train}.tsv")

    res_df.to_csv(res_path, sep="\t", index=False)
    pred_df.to_csv(pred_path, sep="\t", index=False)

    log(f"Summary metrics saved to {res_path}")
    log(f"Per-sample predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
