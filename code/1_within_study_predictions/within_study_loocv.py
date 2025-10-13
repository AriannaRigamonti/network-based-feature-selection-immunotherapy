#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Script: within_study_loocv.py
# Author: Arianna Rigamonti
# Description:
#   Within-study Leave-One-Out Cross-Validation (LOOCV) for immunotherapy
#   response prediction across multiple feature sets (NetBio, Wallenius,
#   GSEA preranked, single-target marker panels) and models
#   (LogisticRegression, RandomForest, SVC). The script performs:
#     - Per-study LOOCV with internal GridSearchCV model selection
#     - Confidence intervals: Wilson (binary metrics) + percentile bootstrap
#       (AUC, PR-AUC, F1, Balanced Accuracy, Macro-F1)
#     - Optional per-fold SHAP value export for model interpretability
#     - Forward feature selection (AIC-driven, statsmodels.Logit) over ALL ssGSEA
#
# Inputs (relative paths expected):
#   - data/immuno/<study>/{study}_mRNA.txt or {study}_mRNA.norm3.txt (Kim)
#   - data/immuno/<study>/{study}_ssGSEA.txt
#   - data/immuno/<study>/{study}_metadata.txt  (columns: Patient, Response[0/1])
#   - data/Marker_summary.txt (columns: Name, Gene_list)
#   - results/0_data_collection_and_preprocessing/<VERSION>/<INTERACTION>/
#       NetBio_original/<study>/<study>_ssGSEA_<target>_NetBio_features.tsv
#       NetBio_wallenius/<study>/<study>_ssGSEA_<target>_Wallenius_features.tsv
#
# Output tree (under results/1_within_study_prediction/{VERSION}_{INTERACTION}/):
#   - summary_metrics.tsv                # aggregate metrics with CIs
#   - fold_predictions.tsv               # per-fold predictions (y_true, y_pred, y_prob)
#   - feature_lists/{Study}/{FeatureSet}_features.tsv
#   - feature_lists/{Study}/feature_counts.tsv
#   - shap/<Study>/<FeatureSet>/<Model>/SHAP_<Sample>.tsv
#   - fs_selected_features/{Study}__FS_AIC(AllSSGSEA)__<Model>__selected.tsv
#   - fs_selected_features/{Study}__FS_AIC(AllSSGSEA)__feature_frequency.tsv
#   - feature_lists/{Study}/FS_AIC(AllSSGSEA)_union_features.tsv
#
# Notes:
#   - Designed for HPC usage (SLURM); honors SLURM_CPUS_PER_TASK for n_jobs.
#   - Reproducibility: seeds are fixed; environment thread caps set for BLAS/OMP.
################################################################################

# ----------------------------- Environment & threading caps -----------------------------
import os
# Limit numerical library threads to avoid oversubscription on HPC nodes
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Temporary directory for joblib (fallback on many clusters)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

# Parallelism for joblib/sklearn: prefer SLURM_CPUS_PER_TASK if available
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

# ----------------------------- Imports -----------------------------
import glob
import time
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans  # used for KernelExplainer background selection

from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
from joblib import Parallel, delayed

import shap  # SHAP interpretability (Tree/Linear/Kernel explainers)

# ----------------------------- Base config -----------------------------

SEED = 42
np.random.seed(SEED)

# Studies and their associated immunotherapy target used to build feature paths
STUDIES = ["Gide", "Liu", "Huang", "Kim", "IMvigor210", "Auslander", "Riaz", "Prat", "PEOPLE", "Ravi"]
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

# Root directory containing study-specific expression/ssGSEA/metadata files
DATA_DIR = "data/immuno"  # expects subfolders per study

# Silence known warnings from CV/statsmodels to keep logs clean on HPC runs
from sklearn.exceptions import UndefinedMetricWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")


# ----------------------------- Confidence intervals -----------------------------
# Utility functions for uncertainty estimates reported in summary_metrics.tsv

def wilson_ci(successes: int, total: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion (accuracy, sensitivity, etc.)."""
    if total == 0:
        return (np.nan, np.nan)
    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return float(low), float(high)

def bootstrap_ci_metric(y_true, y_pred, scorer, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for any metric computed from paired (y_true, y_pred)."""
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
    """Percentile bootstrap CI for ROC-AUC using predicted probabilities."""
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
        if not np.isnan(auc):
            vals.append(auc)
    if len(vals) == 0:
        return (np.nan, np.nan)
    vals = np.array(vals, dtype=float)
    lo = np.percentile(vals, 100*alpha/2)
    hi = np.percentile(vals, 100*(1-alpha/2))
    return float(lo), float(hi)


# ----------------------------- Data loaders -----------------------------
# File-system helpers to load study-specific matrices and labels.

def _expr_path_for_study(study: str) -> str:
    """Return the gene-expression file path, handling Kim's normalized file."""
    base = f"{DATA_DIR}/{study}"
    return os.path.join(base, f"{study}_mRNA.norm3.txt") if study == "Kim" else os.path.join(base, f"{study}_mRNA.txt")

def load_gene_expression(study: str) -> pd.DataFrame:
    """Load gene-level expression (genes x samples). Index upper-cased to match marker lists."""
    p = _expr_path_for_study(study)
    df = pd.read_csv(p, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.upper()
    return df

def load_ssgsea_matrix(study: str) -> pd.DataFrame:
    """Load ssGSEA NES (terms x samples), unfiltered pool for FS and GSEA features."""
    p = os.path.join(DATA_DIR, study, f"{study}_ssGSEA.txt")
    df = pd.read_csv(p, sep="\t", index_col=0)
    return df

def load_labels(study: str) -> pd.Series:
    """Load binary response labels (index: Patient IDs; values in {0,1})."""
    data_dir = DATA_DIR
    fldr = study
    # Allow for study folders with additional suffixes (avoid hard failures)
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
    # Strict check: enforce 0/1 labels only
    vals = pd.unique(s)
    if not set(map(str, vals)).issubset({"0", "1"}):
        raise ValueError(f"{study}: Response should contain only 0/1, found {list(vals)}.")
    return s

# ----------------------------- Marker genes -----------------------------
# Marker panels are read from Marker_summary.txt (Name, Gene_list colon-separated).

MARKER_SUMMARY_PATH = "data/Marker_summary.txt"

def _load_marker_summary():
    """Read Marker_summary and validate schema for target panels."""
    bio_df = pd.read_csv(MARKER_SUMMARY_PATH, sep="\t")
    required_cols = {"Name", "Gene_list"}
    missing = required_cols.difference(bio_df.columns)
    if missing:
        raise ValueError(f"Marker_summary.txt missing columns: {sorted(missing)}")
    bio_df["Name"] = bio_df["Name"].astype(str)
    bio_df["Gene_list"] = bio_df["Gene_list"].astype(str)
    return bio_df

_BIO_DF = None

def marker_genes_for_target(target_name: str) -> list:
    """
    Return a unique, upper-cased list of HGNC symbols for the requested single target
    (e.g., 'PD1', 'PD-L1', 'CTLA4') from Marker_summary.txt, matching 'Name' case-insensitively.
    """
    global _BIO_DF
    if _BIO_DF is None:
        _BIO_DF = _load_marker_summary()

    # Normalize lookup (case-insensitive) on the 'Name' column
    tnorm = str(target_name).strip().upper()
    rows = _BIO_DF.loc[_BIO_DF["Name"].str.upper() == tnorm, "Gene_list"]

    if rows.empty:
        raise ValueError(f"Target '{target_name}' not found in {MARKER_SUMMARY_PATH} (column 'Name').")

    genes: list[str] = []
    for gs in rows.tolist():
        genes.extend([g.strip().upper() for g in gs.split(":") if g.strip()])

    # De-duplicate while preserving lexical order for stable columns
    return sorted(set(genes))


# ----------------------------- Feature builders -----------------------------
# Load different feature sets (samples x features), returning (X_df, feature_list).

def load_gsea_features(study: str, target: str, version: str, interaction: str):
    """Load GSEA preranked features already computed to ssGSEA-like matrix (terms x samples)."""
    p = os.path.join(DATA_DIR, study, f"{study}_ssGSEA_{target}_{version}_{interaction}.txt")
    nes = pd.read_csv(p, sep="\t", index_col=0)
    X = nes.T.apply(pd.to_numeric, errors="coerce")   # ensure numeric dtype
    return X, list(X.columns)

def load_netbio_features(study: str, target: str, fo_base: str):
    """
    Load ssGSEA NES filtered by NetBio original (hypergeometric):
      results/.../<version>/<interaction>/NetBio_original/<study>/<study>_ssGSEA_<target>_NetBio_features.tsv
    """
    netbio_root = os.path.join(fo_base, "NetBio_original")
    pattern = os.path.join(netbio_root, study, f"{study}_ssGSEA_{target}_NetBio_features.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NetBio data for {study}. Expected: {pattern}")
    summ = pd.read_csv(files[0], sep="\t", index_col=0)
    X = summ.T.apply(pd.to_numeric, errors="coerce")
    return X, list(X.columns)

def load_wallenius_features(study: str, target: str, fo_base: str):
    """Load ssGSEA NES filtered by weighted ORA (Wallenius) under NetBio_wallenius directory."""
    w_root = os.path.join(fo_base, "NetBio_wallenius")
    pattern = os.path.join(w_root, study, f"{study}_ssGSEA_{target}_Wallenius_features.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Wallenius data for {study}. Expected: {pattern}")
    summ = pd.read_csv(files[0], sep="\t", index_col=0)
    X = summ.T.apply(pd.to_numeric, errors="coerce")  # float conversion
    return X, list(X.columns)

def load_marker_gene_features(study: str, target_name: str):
    """
    Build a (samples x genes) matrix for a specific immunotherapy target panel
    (PD1, PD-L1, or CTLA4), independent of the study's original target.
    """
    # Load gene-level expression (genes x samples); index upper-cased inside
    expr = load_gene_expression(study)

    # Retrieve the target-specific gene list from Marker_summary.txt
    genes = marker_genes_for_target(target_name)

    # Intersect with available genes in the expression matrix
    genes_in_matrix = [g for g in genes if g in expr.index]
    if len(genes_in_matrix) == 0:
        missing = ", ".join(genes)
        raise ValueError(
            f"{study}: panel '{target_name}' not found in expression matrix. "
            f"Checked genes: {missing}"
        )

    # Return samples x genes
    X = expr.loc[genes_in_matrix].T.copy()
    return X, list(X.columns)


# ----------------------------- Models -----------------------------
# Factory functions that build sklearn Pipelines and param grids for GridSearchCV.

def make_lr_pipeline():
    """Logistic Regression with L2 regularization and z-score scaling."""
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=SEED
        ))
    ])
    grid = {"clf__C": np.round(np.concatenate([np.linspace(0.01, 0.1, 10), np.linspace(0.2, 1.5, 14)]), 3)}
    return pipe, grid

def make_svc_pipeline():
    """Support Vector Classifier (linear/RBF kernels) with probability estimates."""
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
    """Random Forest with a large number of trees; class_weight balanced for imbalance."""
    pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=SEED, n_jobs=1
        ))
    ])
    grid = {
        "clf__max_depth": [None, 3, 5, 7, 9],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    }
    return pipe, grid

# Map human-readable model names to factory callables
MODEL_FACTORIES = {
    "LogisticRegression": make_lr_pipeline,
    "SVC": make_svc_pipeline,
    "RandomForest": make_rf_pipeline,
}


# ----------------------------- SHAP helper -----------------------------
# Compute and persist SHAP values for held-out samples, handling SHAP API variants.

def compute_shap_fold(model_name, best_estimator, X_tr_sel, X_te_sel, feat_names, random_state=SEED):
    """
    Compute SHAP values for the held-out sample in a LOOCV fold.
    Returns a dict with 'features' (list[str]), 'shap_values' (list[float]), 'base_value' (float).
    Handles multiple SHAP output shapes robustly across SHAP versions.
    """
    import shap
    from shap import maskers, links
    import numpy as np

    def _transform_for_model(estimator, X):
        """Apply internal scaler if present in the Pipeline (for consistent SHAP inputs)."""
        if isinstance(estimator, Pipeline) and 'scaler' in estimator.named_steps:
            return estimator.named_steps['scaler'].transform(X)
        return X

    def _as_float(x):
        return np.asarray(x, dtype=np.float64)

    def _pick_shap_vector(shap_vals, n_features, x_test_shape=None):
        """
        Convert SHAP outputs (list/ndarray, various shapes) to a 1D vector of length n_features
        for the positive class if present, otherwise the last class.
        """
        # If SHAP returns a list (one array per class), take the last (typically class 1) and the first sample
        if isinstance(shap_vals, list):
            arr = np.asarray(shap_vals[-1], dtype=float)  # choose positive class if binary
        else:
            arr = np.asarray(shap_vals, dtype=float)

        # Common cases
        if arr.ndim == 1 and arr.size == n_features:
            return arr
        if arr.ndim == 2:
            # (n_samples, n_features) -> take the single sample
            if arr.shape[0] == 1 and arr.shape[1] == n_features:
                return arr[0]
            # (n_classes, n_features) -> take positive/last class
            if arr.shape[1] == n_features:
                cls_idx = 1 if arr.shape[0] >= 2 else 0
                return arr[cls_idx]
        if arr.ndim == 3:
            # Possible layouts: (n_samples, n_classes, n_features) or (n_samples, n_features, n_classes)
            # Assume first dim is sample (=1 in our LOOCV)
            if arr.shape[0] == 1:
                # (1, n_classes, n_features)
                if arr.shape[2] == n_features:
                    cls_idx = 1 if arr.shape[1] >= 2 else 0
                    return arr[0, cls_idx, :]
                # (1, n_features, n_classes)
                if arr.shape[1] == n_features:
                    cls_idx = 1 if arr.shape[2] >= 2 else 0
                    return arr[0, :, cls_idx]
        # Fallback: find an axis equal to n_features and slice it
        for axis in range(arr.ndim):
            if arr.shape[axis] == n_features:
                index = [0] * arr.ndim
                index[axis] = slice(None)
                vec = arr[tuple(index)]
                return np.asarray(vec, dtype=float).ravel()
        raise ValueError(f"Unexpected SHAP shape {arr.shape} vs n_features={n_features}")

    # Prepare matrices (respect pipeline scaling when present)
    X_train_m = _as_float(_transform_for_model(best_estimator, X_tr_sel))
    X_test_m  = _as_float(_transform_for_model(best_estimator, X_te_sel))
    n_features = X_train_m.shape[1]

    # Extract the classifier if a Pipeline is passed
    clf = best_estimator
    if isinstance(best_estimator, Pipeline):
        clf = best_estimator.named_steps['clf']

    # Choose suitable explainer per model family
    if model_name == "RandomForest":
        # TreeExplainer for tree ensembles
        explainer = shap.TreeExplainer(clf, data=X_train_m)
        shap_vals = explainer.shap_values(X_test_m)
        sv = _pick_shap_vector(shap_vals, n_features, x_test_shape=X_test_m.shape)
        # expected_value can be scalar or per-class
        ev = np.array(explainer.expected_value, dtype=float)
        base_val = float(ev[1] if ev.ndim > 0 and ev.size >= 2 else ev.ravel()[0])

    elif model_name == "LogisticRegression":
        # LinearExplainer with logit link for logistic models
        explainer = shap.LinearExplainer(
            clf,
            maskers.Independent(X_train_m),
            link=links.logit
        )
        shap_vals = explainer.shap_values(X_test_m)
        sv = _pick_shap_vector(shap_vals, n_features, x_test_shape=X_test_m.shape)
        base_val = float(np.array(explainer.expected_value).ravel()[0])

    elif model_name == "SVC":
        # Linear: LinearExplainer; RBF: KernelExplainer with KMeans background
        if getattr(clf, "kernel", None) == "linear":
            explainer = shap.LinearExplainer(
                clf,
                maskers.Independent(X_train_m),
                link=links.identity
            )
            shap_vals = explainer.shap_values(X_test_m)
            sv = _pick_shap_vector(shap_vals, n_features, x_test_shape=X_test_m.shape)
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
            sv = _pick_shap_vector(shap_vals, n_features, x_test_shape=X_te_raw.shape)
            base_val = float(np.array(explainer.expected_value).ravel()[0])

    else:
        raise ValueError(f"Unsupported model for SHAP: {model_name}")

    return {
        "features": list(feat_names),
        "shap_values": np.asarray(sv, dtype=float).ravel().tolist(),
        "base_value": float(base_val),
    }


# ----------------------------- LOOCV core -----------------------------
# Core routine to run LOOCV with internal hyperparameter search and optional SHAP.

def run_loocv_single(study, X_df, y_vec, model_name="LogisticRegression",
                     feat_name=None, shap_root=None):
    """
    LOOCV with internal GridSearchCV; returns per-fold predictions and metrics with CIs.
    If shap_root is provided, saves per-fold SHAP TSVs under:
      {shap_root}/{study}/{feat_name}/{model_name}/SHAP_<Sample>.tsv
    """
    # Align samples and ensure numeric dtype
    if isinstance(y_vec, pd.Series):
        common = X_df.index.intersection(y_vec.index)
        X = X_df.loc[common].values
        y = y_vec.loc[common].astype(int).values
        samples = list(common)
    else:
        X = X_df.values
        y = np.asarray(y_vec).astype(int)
        samples = list(X_df.index)

    if len(np.unique(y)) < 2:
        raise ValueError(f"{study}: only one class present in labels.")

    loo = LeaveOneOut()
    preds, probs, obs, test_samples = [], [], [], []

    # Build model and param grid
    pipe, grid = MODEL_FACTORIES[model_name]()
    feat_names = list(X_df.columns)

    # LOOCV loop with GridSearchCV on the training fold
    for train_idx, test_idx in loo.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        gcv = GridSearchCV(
            estimator=pipe, param_grid=grid, scoring="roc_auc",
            cv=5 if len(np.unique(y_tr)) > 1 and len(y_tr) >= 5 else 3,
            n_jobs=N_JOBS, refit=True
        )
        gcv.fit(X_tr, y_tr)

        # Predict label and probability for the held-out sample
        y_hat = gcv.best_estimator_.predict(X_te)[0]
        try:
            y_pr = gcv.best_estimator_.predict_proba(X_te)[0, 1]
        except Exception:
            # Fallback: map decision_function to [0,1] via logistic transform
            if hasattr(gcv.best_estimator_, "decision_function"):
                s = gcv.best_estimator_.decision_function(X_te)
                y_pr = 1.0/(1.0 + np.exp(-float(s)))
            else:
                y_pr = float(y_hat)

        preds.append(int(y_hat))
        probs.append(float(y_pr))
        obs.append(int(y_te[0]))
        test_samples.append(samples[test_idx[0]])

        # --------- SHAP save ----------
        if shap_root is not None and feat_name is not None:
            try:
                shap_out = compute_shap_fold(
                    model_name=model_name,
                    best_estimator=gcv.best_estimator_,
                    X_tr_sel=X_tr,            # no FS here: full columns
                    X_te_sel=X_te,
                    feat_names=feat_names
                )
                shap_dir = os.path.join(shap_root, study, str(feat_name), model_name)
                os.makedirs(shap_dir, exist_ok=True)
                assert len(shap_out["features"]) == len(shap_out["shap_values"]), (
                f"SHAP length mismatch: features={len(shap_out['features'])} "
                f"vs shap_values={len(shap_out['shap_values'])}"
                )
                shap_df = pd.DataFrame({
                    "Feature": shap_out["features"],
                    "SHAP": shap_out["shap_values"]
                })
                shap_df["Sample"] = samples[test_idx[0]]
                shap_path = os.path.join(shap_dir, f"SHAP_{samples[test_idx[0]]}.tsv")
                shap_df.to_csv(shap_path, sep="\t", index=False)
            except Exception as e:
                print(f"        [SHAP WARN] {feat_name} × {model_name} (sample={samples[test_idx[0]]}): {e}")
        # -----------------------------------

    # Convert to arrays for metric computation
    obs = np.array(obs, dtype=int)
    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)

    # Confusion matrix components and base metrics
    tn, fp, fn, tp = confusion_matrix(obs, preds, labels=[0, 1]).ravel()
    acc  = accuracy_score(obs, preds)
    prec = precision_score(obs, preds, zero_division=0)
    rec  = recall_score(obs, preds, zero_division=0)
    f1   = f1_score(obs, preds, zero_division=0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    bal_acc = balanced_accuracy_score(obs, preds)
    f1_macro = f1_score(obs, preds, average="macro", zero_division=0)

    # Threshold-free metrics
    try:
        auc = roc_auc_score(obs, probs)
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score(obs, probs)
    except Exception:
        ap = np.nan

    # Confidence intervals (Wilson for binomial; bootstrap for others)
    acc_ci  = wilson_ci(tp + tn, len(obs))
    sens_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (np.nan, np.nan)
    spec_ci = wilson_ci(tn, tn + fp) if (tn + fp) > 0 else (np.nan, np.nan)
    prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (np.nan, np.nan)
    f1_ci   = bootstrap_ci_metric(obs, preds, lambda y_t, y_p: f1_score(y_t, y_p))
    auc_ci  = bootstrap_ci_auc(obs, probs) if np.isfinite(auc) else (np.nan, np.nan)
    ap_ci   = bootstrap_ci_metric(obs, probs, lambda y_t, p: average_precision_score(y_t, p)) if np.isfinite(ap) else (np.nan, np.nan)
    bal_acc_ci = bootstrap_ci_metric(obs, preds,
                                     lambda y_t, y_p: balanced_accuracy_score(y_t, y_p))
    f1_macro_ci = bootstrap_ci_metric(obs, preds,
                                      lambda y_t, y_p: f1_score(y_t, y_p, average="macro", zero_division=0))

    return {
        "study": study,
        "samples": test_samples,
        "y_true": obs,
        "y_pred": preds,
        "y_prob": probs,
        "metrics": {
            "accuracy": (acc, *acc_ci),
            "balanced_accuracy": (bal_acc, *bal_acc_ci),
            "precision": (prec, *prec_ci),
            "recall": (rec, *sens_ci),
            "specificity": (spec, *spec_ci),
            "f1": (f1, *f1_ci),
            "f1_macro": (f1_macro, *f1_macro_ci),
            "roc_auc": (auc, *auc_ci),
            "pr_auc": (ap, *ap_ci)
        }
    }


# ----------------------------- Forward Selection on ALL-ssGSEA -----------------------------
# Greedy forward AIC selection (statsmodels.Logit) evaluated inside each training fold.

FS_DELTA  = 0.01   # minimum AIC improvement to continue
FS_NJOBS  = 10     # parallel jobs per forward step

def _fit_logit_and_aic(X_sm, y) -> float:
    """Fit a logistic regression (statsmodels) and return AIC; inf on failure."""
    try:
        res = sm.Logit(y, X_sm).fit(method="lbfgs", disp=False, maxiter=500)
        return float(res.aic) if res.aic is not None else np.inf
    except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError, ValueError):
        return np.inf

def _forward_step_aic(X, y, selected, feat) -> Tuple[int, float]:
    """Single forward step candidate evaluation: returns (feature_index, AIC)."""
    idx = selected + [feat]
    aic_val = _fit_logit_and_aic(sm.add_constant(X[:, idx]), y)
    return feat, aic_val

def forward_sfs_aic(X: np.ndarray, y: np.ndarray,
                    delta: float = 1e-2, n_jobs: int = 1,
                    return_trace: bool = False):
    """
    Greedy forward selection driven by AIC.
    Stops when AIC improvement is < delta.
    """
    selected, remaining = [], list(range(X.shape[1]))
    last_aic = np.inf; trace: List[Tuple[int, float]] = []
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

def run_loocv_fs_ssgsea_all(study, y_series, model_name="LogisticRegression",
                            delta: float = FS_DELTA, n_jobs: int = FS_NJOBS,
                            shap_root=None):
    """
    LOOCV + forward AIC-based feature selection over the full ssGSEA feature space.
    Pipeline:
      - z-score on training only (for FS step)
      - forward SFS via statsmodels.Logit (AIC) on training folds
      - fit sklearn model (with its own scaling) on the selected columns
      - evaluate on the left-out sample
      - compute per-fold SHAP on the selected columns (if shap_root is provided)
    """
    nes = load_ssgsea_matrix(study)  # terms x samples
    X_full = nes.T  # samples x pathways

    # Ensure enough samples and both classes present
    common = X_full.index.intersection(y_series.index)
    if len(common) < 3 or len(np.unique(y_series.loc[common])) < 2:
        raise ValueError(f"{study}: too few samples or one class for FS.")
    X_df = X_full.loc[common].copy()
    y = y_series.loc[common].astype(int).values
    samples = list(X_df.index)
    feat_names = list(X_df.columns)

    X_all = X_df.values
    loo = LeaveOneOut()

    preds, probs, obs, test_samples = [], [], [], []
    sel_features_per_fold = []

    pipe, grid = MODEL_FACTORIES[model_name]()
    fs_name = "FS_AIC(AllssGSEA)"

    for tr_idx, te_idx in loo.split(X_all, y):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # scale training fold for FS only (avoid leakage)
        scaler_fs = StandardScaler(with_mean=True, with_std=True).fit(X_tr)
        X_tr_z = scaler_fs.transform(X_tr)

        # forward AIC selection on training
        sel_idx, trace = forward_sfs_aic(
            X_tr_z, y_tr, delta=delta, n_jobs=n_jobs, return_trace=True
        )

        # choose the first step within +1.0 AIC of the minimum (more robust than strict argmin)
        if trace:
            steps, aics = zip(*trace)
            best_aic = float(np.min(aics)); tol = 1.0
            best_step = next(s for s, a in trace if a <= best_aic + tol)
        else:
            best_step = len(sel_idx) if len(sel_idx) > 0 else 0

        sel_idx = sel_idx[:best_step] if best_step > 0 else sel_idx
        if len(sel_idx) == 0:
            # ensure at least one feature is used if selection fails
            single = forward_sfs_aic(X_tr_z, y_tr, delta=delta, n_jobs=n_jobs, return_trace=False)
            sel_idx = single if isinstance(single, list) and single else [0]

        sel_feats = [feat_names[i] for i in sel_idx]
        sel_features_per_fold.append("|".join(sel_feats))

        # train sklearn model on selected columns (pipeline handles scaling internally)
        X_tr_sel = X_tr[:, sel_idx]
        X_te_sel = X_te[:, sel_idx]
        X_te_sel = np.atleast_2d(X_te_sel)

        gcv = GridSearchCV(
            estimator=pipe, param_grid=grid, scoring="roc_auc",
            cv=5 if len(np.unique(y_tr)) > 1 and len(y_tr) >= 5 else 3,
            n_jobs=N_JOBS, refit=True
        )
        gcv.fit(X_tr_sel, y_tr)

        # Held-out predictions
        y_hat = gcv.best_estimator_.predict(X_te_sel)[0]
        try:
            y_pr = gcv.best_estimator_.predict_proba(X_te_sel)[0, 1]
        except Exception:
            if hasattr(gcv.best_estimator_, "decision_function"):
                s = gcv.best_estimator_.decision_function(X_te_sel)
                y_pr = 1.0/(1.0 + np.exp(-float(s)))
            else:
                y_pr = float(y_hat)

        preds.append(int(y_hat)); probs.append(float(y_pr))
        obs.append(int(y_te[0])); test_samples.append(samples[te_idx[0]])

        # --------- SHAP save (on selected columns) ----------
        if shap_root is not None:
            try:
                shap_out = compute_shap_fold(
                    model_name=model_name,
                    best_estimator=gcv.best_estimator_,
                    X_tr_sel=X_tr_sel,
                    X_te_sel=X_te_sel,
                    feat_names=sel_feats
                )
                shap_dir = os.path.join(shap_root, study, fs_name, model_name)
                os.makedirs(shap_dir, exist_ok=True)
                shap_df = pd.DataFrame({
                    "Feature": shap_out["features"],
                    "SHAP": shap_out["shap_values"]
                })
                shap_df["Sample"] = samples[te_idx[0]]
                shap_path = os.path.join(shap_dir, f"SHAP_{samples[te_idx[0]]}.tsv")
                shap_df.to_csv(shap_path, sep="\t", index=False)
            except Exception as e:
                print(f"        [SHAP WARN] {fs_name} × {model_name} (sample={samples[te_idx[0]]}): {e}")
        # ---------------------------------------------------------

    # Compute metrics + CIs for FS run
    obs = np.array(obs, dtype=int); preds = np.array(preds, dtype=int); probs = np.array(probs, dtype=float)
    tn, fp, fn, tp = confusion_matrix(obs, preds, labels=[0, 1]).ravel()
    acc, prec = accuracy_score(obs, preds), precision_score(obs, preds, zero_division=0)
    rec, f1   = recall_score(obs, preds, zero_division=0), f1_score(obs, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(obs, preds)
    f1_macro = f1_score(obs, preds, average="macro", zero_division=0)
    sens = tp/(tp+fn) if (tp+fn)>0 else np.nan
    spec = tn/(tn+fp) if (tn+fp)>0 else np.nan
    try: auc = roc_auc_score(obs, probs)
    except Exception: auc = np.nan
    try: ap = average_precision_score(obs, probs)
    except Exception: ap = np.nan

    acc_ci  = wilson_ci(tp + tn, len(obs))
    bal_acc_ci = bootstrap_ci_metric(obs, preds,
                                     lambda y_t, y_p: balanced_accuracy_score(y_t, y_p))
    sens_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (np.nan, np.nan)
    spec_ci = wilson_ci(tn, tn + fp) if (tn + fp) > 0 else (np.nan, np.nan)
    prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (np.nan, np.nan)
    f1_ci   = bootstrap_ci_metric(obs, preds, lambda y_t, y_p: f1_score(y_t, y_p))
    f1_macro_ci = bootstrap_ci_metric(obs, preds,
                                      lambda y_t, y_p: f1_score(y_t, y_p, average="macro", zero_division=0))
    auc_ci  = bootstrap_ci_auc(obs, probs) if np.isfinite(auc) else (np.nan, np.nan)
    ap_ci   = bootstrap_ci_metric(obs, probs, lambda y_t, p: average_precision_score(y_t, p)) if np.isfinite(ap) else (np.nan, np.nan)

    return {
        "study": study,
        "samples": test_samples,
        "y_true": obs, "y_pred": preds, "y_prob": probs,
        "selected_features": sel_features_per_fold,
        "metrics": {
            "accuracy": (acc, *acc_ci),
            "balanced_accuracy": (bal_acc, *bal_acc_ci),
            "precision": (prec, *prec_ci),
            "recall": (rec, *sens_ci),
            "specificity": (spec, *spec_ci),
            "f1": (f1, *f1_ci),
            "f1_macro": (f1_macro, *f1_macro_ci),
            "roc_auc": (auc, *auc_ci),
            "pr_auc": (ap, *ap_ci)
        }
    }


# ----------------------------- Helper: stratified subsampling -----------------------------
# Utility to select a smaller, roughly balanced subset of patients per study.

def stratified_take(y: pd.Series, n: int, random_state: int = SEED):
    """
    Sample up to n patients trying to keep classes balanced.
    Ensures at least 1 per class if both classes exist and counts allow it.
    """
    y = y.dropna().astype(int)
    if len(y) == 0:
        return []

    cls = y.value_counts()
    if len(cls) < 2:
        return y.index.tolist()[:min(n, len(y))]

    n_pos_target = n // 2
    n_neg_target = n - n_pos_target

    n_pos = int(min(cls.get(1, 0), n_pos_target))
    n_neg = int(min(cls.get(0, 0), n_neg_target))

    if n_pos == 0 and cls.get(1, 0) > 0:
        n_pos = 1
    if n_neg == 0 and cls.get(0, 0) > 0:
        n_neg = 1

    short = n - (n_pos + n_neg)
    if short > 0:
        if cls.get(1, 0) - n_pos >= cls.get(0, 0) - n_neg:
            n_pos = int(min(cls.get(1, 0), n_pos + short))
        else:
            n_neg = int(min(cls.get(0, 0), n_neg + short))

    rng = np.random.default_rng(random_state)
    pos_ids_all = np.array(y.index[y == 1])
    neg_ids_all = np.array(y.index[y == 0])

    pos_pick = rng.choice(pos_ids_all, n_pos, replace=False) if n_pos > 0 else np.array([], dtype=object)
    neg_pick = rng.choice(neg_ids_all, n_neg, replace=False) if n_neg > 0 else np.array([], dtype=object)

    sel = np.concatenate([pos_pick, neg_pick]).tolist()
    rng.shuffle(sel)
    return sel


# ----------------------------- MAIN RUNNER -----------------------------
# CLI entry point: selects study, models, FS parameters, and manages outputs.

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Within-study LOOCV")
    parser.add_argument("--version", default="v11", choices=["v11", "v12"])
    parser.add_argument("--interaction", default="full", choices=["full", "physical"])
    parser.add_argument("--study", required=True,
                        help="One of: " + " ".join(STUDIES))
    parser.add_argument("--n_patients", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForest", "SVC"])
    parser.add_argument("--fs_delta", type=float, default=0.01) 
    parser.add_argument("--fs_njobs", type=int, default=8)
    parser.add_argument("--out_root", default="results/1_within_study_prediction")
    args = parser.parse_args()

    # Log effective threading/BLAS environment for provenance
    print(
    f"[INFO] N_JOBS={N_JOBS} | SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')} | "
    f"OMP={os.environ.get('OMP_NUM_THREADS')} | OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} | "
    f"MKL={os.environ.get('MKL_NUM_THREADS')} | NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}"
    )

    # Update FS config from CLI
    global FS_DELTA, FS_NJOBS
    FS_DELTA = args.fs_delta
    FS_NJOBS = args.fs_njobs

    VERSION     = args.version
    INTERACTION = args.interaction
    FO_BASE     = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"
    FO_ML       = f"{args.out_root}/{VERSION}_{INTERACTION}"
    os.makedirs(FO_ML, exist_ok=True)

    # SHAP root directory
    SHAP_ROOT   = os.path.join(FO_ML, "shap")
    os.makedirs(SHAP_ROOT, exist_ok=True)

    # Directories for feature lists and counts
    FEAT_LISTS_ROOT = os.path.join(FO_ML, "feature_lists")
    os.makedirs(FEAT_LISTS_ROOT, exist_ok=True)

    # Helper: append-or-write a single-row TSV (used for counts)
    def _append_or_write_row(row_dict, path):
        row_df = pd.DataFrame([row_dict])
        if os.path.exists(path):
            old = pd.read_csv(path, sep="\t")
            pd.concat([old, row_df], axis=0, ignore_index=True).to_csv(path, sep="\t", index=False)
        else:
            row_df.to_csv(path, sep="\t", index=False)

    # Feature builders with resolved paths/params
    def _fb_netbio(st):
        return load_netbio_features(st, TARGET_BY_STUDY[st], FO_BASE)
    def _fb_wallenius(st):
        return load_wallenius_features(st, TARGET_BY_STUDY[st], FO_BASE)
    def _fb_gsea(st):
        return load_gsea_features(st, TARGET_BY_STUDY[st], VERSION, INTERACTION)
    def _fb_pd1(st):
        return load_marker_gene_features(st, "PD1")
    def _fb_pdl1(st):
        return load_marker_gene_features(st, "PD-L1")
    def _fb_ctla4(st):
        return load_marker_gene_features(st, "CTLA4")

    feature_builders = {
        "NetBio":            _fb_netbio,
        "NetBio_Wallenius":  _fb_wallenius,
        "GSEA":              _fb_gsea,
        "PD1":               _fb_pd1,
        "PD-L1":             _fb_pdl1,
        "CTLA4":             _fb_ctla4,
}

    models = args.models

    # Limit to the requested study (one study per job for HPC)
    studies_to_run = [args.study]

    rows = []
    pred_rows = []

    FS_SEL_DIR = os.path.join(FO_ML, "fs_selected_features")
    os.makedirs(FS_SEL_DIR, exist_ok=True)

    t0_all = time.time()
    print(f"[START] {time.ctime(t0_all)} | VERSION={VERSION} INTERACTION={INTERACTION} | subset per study = {args.n_patients}")

    for study in studies_to_run:
        t_study = time.time()
        print("\n" + "="*80)
        print(f"[{time.ctime()}] Study: {study}")

        # Labels
        try:
            y_series_full = load_labels(study)
            n_pos = int((y_series_full == 1).sum())
            n_neg = int((y_series_full == 0).sum())
            print(f"  · labels loaded: total={len(y_series_full)} (pos={n_pos}, neg={n_neg})")
        except Exception as e:
            print(f"[WARN] {study} / labels: {e}")
            continue

        # Subset selection (reused across all feature sets + FS)
        if args.n_patients is not None and args.n_patients > 0:
            sel_ids = stratified_take(y_series_full, args.n_patients, random_state=SEED)
            y_series = y_series_full.loc[sel_ids]
            print(f"  · subset: n={len(y_series)}  (pos={(y_series==1).sum()}, neg={(y_series==0).sum()})")
            if len(np.unique(y_series)) < 2 or len(y_series) < 3:
                print(f"[WARN] {study}: subset has too few samples or a single class. Skipping.")
                continue
        else:
            y_series = y_series_full
            sel_ids = y_series.index.tolist()

        # =============== Standard feature sets (NetBio/Wallenius/GSEA/Markers) ===============
        for feat_name, builder in feature_builders.items():
            print(f"  -> Loading features: {feat_name}")
            try:
                X_df, feat_list = builder(study)
            except Exception as e:
                print(f"[WARN] {study} / {feat_name}: {e}")
                continue

            # Align to subset and check classes (alignment does not change columns)
            common = X_df.index.intersection(y_series.index)
            if len(common) < 3 or len(np.unique(y_series.loc[common])) < 2:
                print(f"[WARN] {study} / {feat_name}: too few samples or one class after align (n={len(common)}).")
                continue
            X_df = X_df.loc[common].copy()
            y = y_series.loc[common].copy()

            # --- Save feature list and count for this (Study, FeatureSet) ---
            feat_dir = os.path.join(FEAT_LISTS_ROOT, study)
            os.makedirs(feat_dir, exist_ok=True)
            flist_path = os.path.join(feat_dir, f"{feat_name}_features.tsv")
            if not os.path.exists(flist_path):
                pd.DataFrame({"Feature": X_df.columns}).to_csv(flist_path, sep="\t", index=False)
            fcount_path = os.path.join(feat_dir, "feature_counts.tsv")
            _append_or_write_row(
                {"Study": study, "Features": feat_name, "n_features": int(X_df.shape[1])},
                fcount_path
            )
            # -----------------------------------------------------------------

            print(f"     · aligned: X={X_df.shape}  y={{1:{int(y.sum())}, 0:{int((1-y).sum())}}}  #features={X_df.shape[1]}")

            for mdl in models:
                print(f"     => LOOCV {feat_name} × {mdl} ...")
                t_model = time.time()
                try:
                    res = run_loocv_single(
                        study, X_df, y, model_name=mdl,
                        feat_name=feat_name, shap_root=SHAP_ROOT
                    )
                except Exception as e:
                    print(f"        [FAIL] {feat_name} × {mdl}: {e}")
                    continue
                dt = time.time() - t_model
                met = res["metrics"]

                print(f"        done in {dt:.1f}s | folds={len(res['y_true'])} | "
                      f"Acc={met['accuracy'][0]:.3f} BalAcc={met['balanced_accuracy'][0]:.3f} AUC={met['roc_auc'][0]:.3f} F1={met['f1'][0]:.3f} F1m={met['f1_macro'][0]:.3f}")

                # Aggregate metrics row
                rows.append({
                    "Study": study,
                    "Features": feat_name,
                    "Model": mdl,
                    "n_samples": len(res["y_true"]),
                    "n_features": X_df.shape[1],
                    "Accuracy": met["accuracy"][0],
                    "Acc_CI_low": met["accuracy"][1],
                    "Acc_CI_high": met["accuracy"][2],
                    "BalancedAcc": met["balanced_accuracy"][0],
                    "BalAcc_CI_low": met["balanced_accuracy"][1],
                    "BalAcc_CI_high": met["balanced_accuracy"][2],
                    "F1": met["f1"][0],
                    "F1_CI_low": met["f1"][1],
                    "F1_CI_high": met["f1"][2],
                    "F1_macro": met["f1_macro"][0],
                    "F1m_CI_low": met["f1_macro"][1],
                    "F1m_CI_high": met["f1_macro"][2],
                    "Sensitivity": met["recall"][0],
                    "Sens_CI_low": met["recall"][1],
                    "Sens_CI_high": met["recall"][2],
                    "Specificity": met["specificity"][0],
                    "Spec_CI_low": met["specificity"][1],
                    "Spec_CI_high": met["specificity"][2],
                    "Precision": met["precision"][0],
                    "Prec_CI_low": met["precision"][1],
                    "Prec_CI_high": met["precision"][2],
                    "ROC_AUC": met["roc_auc"][0],
                    "ROC_AUC_CI_low": met["roc_auc"][1],
                    "ROC_AUC_CI_high": met["roc_auc"][2],
                    "PR_AUC": met["pr_auc"][0],
                    "PR_AUC_CI_low": met["pr_auc"][1],
                    "PR_AUC_CI_high": met["pr_auc"][2],
                })

                # Per-fold predictions
                for s, yt, yp, pr in zip(res["samples"], res["y_true"], res["y_pred"], res["y_prob"]):
                    pred_rows.append({
                        "Study": study,
                        "Features": feat_name,
                        "Model": mdl,
                        "Sample": s,
                        "y_true": int(yt),
                        "y_pred": int(yp),
                        "y_prob": float(pr)
                    })

        # ================= Forward Selection over ALL-ssGSEA =================
        try:
            pool_n = load_ssgsea_matrix(study).shape[0]
        except Exception:
            pool_n = np.nan
        print(f"  -> FS_AIC over ALL ssGSEA (candidate pool ~ {pool_n})")

        for mdl in models:
            fs_name = "FS_AIC(AllSSGSEA)"
            print(f"     => FS+LOOCV {fs_name} × {mdl} ...")
            t_fs = time.time()
            try:
                res_fs = run_loocv_fs_ssgsea_all(
                    study, y_series.loc[sel_ids], model_name=mdl,
                    delta=FS_DELTA, n_jobs=FS_NJOBS,
                    shap_root=SHAP_ROOT
                )
            except Exception as e:
                print(f"        [FAIL] {fs_name} × {mdl}: {e}")
                continue
            dt = time.time() - t_fs
            met = res_fs["metrics"]
            print(f"        done in {dt:.1f}s | folds={len(res_fs['y_true'])} | "
                  f"Acc={met['accuracy'][0]:.3f} AUC={met['roc_auc'][0]:.3f} F1={met['f1'][0]:.3f}")

            rows.append({
                "Study": study,
                "Features": fs_name,
                "Model": mdl,
                "n_samples": len(res_fs["y_true"]),
                "n_features": pool_n,
                "Accuracy": met["accuracy"][0],
                "Acc_CI_low": met["accuracy"][1],
                "Acc_CI_high": met["accuracy"][2],
                "BalancedAcc": met["balanced_accuracy"][0],
                "BalAcc_CI_low": met["balanced_accuracy"][1],
                "BalAcc_CI_high": met["balanced_accuracy"][2],
                "F1": met["f1"][0],
                "F1_CI_low": met["f1"][1],
                "F1_CI_high": met["f1"][2],
                "F1_macro": met["f1_macro"][0],
                "F1m_CI_low": met["f1_macro"][1],
                "F1m_CI_high": met["f1_macro"][2],
                "Sensitivity": met["recall"][0],
                "Sens_CI_low": met["recall"][1],
                "Sens_CI_high": met["recall"][2],
                "Specificity": met["specificity"][0],
                "Spec_CI_low": met["specificity"][1],
                "Spec_CI_high": met["specificity"][2],
                "Precision": met["precision"][0],
                "Prec_CI_low": met["precision"][1],
                "Prec_CI_high": met["precision"][2],
                "ROC_AUC": met["roc_auc"][0],
                "ROC_AUC_CI_low": met["roc_auc"][1],
                "ROC_AUC_CI_high": met["roc_auc"][2],
                "PR_AUC": met["pr_auc"][0],
                "PR_AUC_CI_low": met["pr_auc"][1],
                "PR_AUC_CI_high": met["pr_auc"][2],
            })

            # Save per-fold selected features
            df_sel = pd.DataFrame({
                "Study": study,
                "Features": fs_name,
                "Model": mdl,
                "Sample": res_fs["samples"],
                "Selected": res_fs["selected_features"],
                "n_selected": [len(s.split("|")) if isinstance(s, str) and len(s) else 0
                               for s in res_fs["selected_features"]],
            })
            out_sel = os.path.join(FS_SEL_DIR, f"{study}__{fs_name}__{mdl}__selected.tsv")
            df_sel.to_csv(out_sel, sep="\t", index=False)
            print(f"        · saved per-fold selections -> {out_sel}")

            # Aggregate selection frequency across folds (FS_AIC)
            try:
                agg = (
                    df_sel.assign(Feature=df_sel["Selected"].str.split("|"))
                         .explode("Feature")
                )
                agg = agg[agg["Feature"].fillna("").str.len() > 0]
                agg = (agg.groupby("Feature").size()
                           .sort_values(ascending=False)
                           .rename("Selected_in_folds")
                           .reset_index())
                agg_path = os.path.join(FS_SEL_DIR, f"{study}__{fs_name}__feature_frequency.tsv")
                agg.to_csv(agg_path, sep="\t", index=False)
                print(f"        · saved FS feature frequency -> {agg_path}")

                # Save union of selected features at least once
                union_feats = agg["Feature"].tolist()
                feat_dir = os.path.join(FEAT_LISTS_ROOT, study)
                os.makedirs(feat_dir, exist_ok=True)
                union_path = os.path.join(feat_dir, f"{fs_name}_union_features.tsv")
                pd.DataFrame({"Feature": union_feats}).to_csv(union_path, sep="\t", index=False)

                # Append count for FS union
                fcount_path = os.path.join(feat_dir, "feature_counts.tsv")
                _append_or_write_row(
                    {"Study": study, "Features": f"{fs_name}_union", "n_features": int(len(union_feats))},
                    fcount_path
                )
            except Exception as e:
                print(f"        [WARN] could not compute FS feature frequency/union: {e}")

            # Per-fold predictions for FS run
            for s, yt, yp, pr in zip(res_fs["samples"], res_fs["y_true"], res_fs["y_pred"], res_fs["y_prob"]):
                pred_rows.append({
                    "Study": study,
                    "Features": fs_name,
                    "Model": mdl,
                    "Sample": s,
                    "y_true": int(yt),
                    "y_pred": int(yp),
                    "y_prob": float(pr)
                })

        print(f"[{study}] completed in {(time.time()-t_study)/60:.1f} min")

    # Final saves (append if files already exist, to allow one-job-per-study accumulation)
    res_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)

    res_path = os.path.join(FO_ML, "summary_metrics.tsv")
    pred_path = os.path.join(FO_ML, "fold_predictions.tsv")

    def _append_or_write(df, path):
        """Append to existing TSV or create it if missing (support multi-job accumulation)."""
        if df is None or df.empty:
            return
        if os.path.exists(path):
            old = pd.read_csv(path, sep="\t")
            df_all = pd.concat([old, df], axis=0, ignore_index=True)
            df_all.to_csv(path, sep="\t", index=False)
        else:
            df.to_csv(path, sep="\t", index=False)

    _append_or_write(res_df, res_path)
    _append_or_write(pred_df, pred_path)

    print("\n" + "-"*80)
    print(f"[DONE] Saved metrics -> {res_path}")
    print(f"[DONE] Saved folds   -> {pred_path}")
    print(f"[TOTAL] Elapsed {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
