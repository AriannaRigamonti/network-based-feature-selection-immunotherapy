################################################################################
# Script: 2b_Wallenius.py
# Author: Arianna Rigamonti
# Description: Weighted ORA (Wallenius noncentral hypergeometric) applied to
#              propagation-based gene scores from NetBio. Reproduces the original
#              pipeline structure but integrates propagation-derived weights.
# Output: For each study, pathway-level results (ORA table, selected terms,
#         ssGSEA features) under results/.../NetBio_wallenius.
################################################################################

import os
os.chdir("path_to_your_repo")  # CHANGE to your local repo path
print(os.getcwd())

import os, glob, time, re
import numpy as np
import pandas as pd
import gseapy as gp
from scipy.stats import nchypergeom_wallenius
from statsmodels.stats.multitest import multipletests
from scipy import stats as stat 

# ----------------------- Configuration -----------------------
# Weighted ORA (Wallenius) reproducing NetBio I/O structure under /NetBio_wallenius
STUDIES = ["Gide", "Liu", "Huang", "Kim", "IMvigor210", "Auslander", "Riaz", "Prat", "PEOPLE", "Ravi"]
TARGET_BY_STUDY = {
    "Gide": "PD1_CTLA4", "Liu": "PD1", "Huang": "PD1", "Kim": "PD1",
    "IMvigor210": "PD-L1", "Auslander": "PD1_CTLA4", "Riaz": "PD1",
    "Prat": "PD1", "PEOPLE": "PD1", "Ravi": "PD1_PD-L1_CTLA4",
}

VERSION      = "v12"             
INTERACTION  = "full"        # "full" or "physical"
PROP_DIR     = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"
TOP_N        = 200               # number of top genes (NetBio convention)
QVAL_CUTOFF  = 0.01              # same FDR threshold as original NetBio
FO_ROOT_W    = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}/NetBio_wallenius"
os.makedirs(FO_ROOT_W, exist_ok=True)

# ----------------------- Reactome pathways -----------------------
def load_reactome_dict():
    # Load Reactome 2024 gene sets via gseapy; output dict term -> list of genes (uppercased)
    lib = gp.get_library(name="Reactome_Pathways_2024", organism="Human")
    if isinstance(lib, dict):
        reactome = {term: [str(g).upper() for g in genes] for term, genes in lib.items()}
    else:
        df = gp.get_library(name="Reactome_Pathways_2024", organism="Human", return_dataframe=True)
        cols = {c.lower(): c for c in df.columns}
        term_col = cols.get("term") or cols.get("name") or list(df.columns)[0]
        gene_col = cols.get("gene_symbol") or cols.get("genes") or list(df.columns)[1]
        reactome = (df.rename(columns={term_col: "Term", gene_col: "Gene"})
                      .groupby("Term")["Gene"].apply(lambda s: [str(x).upper() for x in s]).to_dict())
    reactome = {k: sorted(set(v)) for k, v in reactome.items() if len(v) > 0}
    if not reactome:
        raise RuntimeError("Empty Reactome library.")
    return reactome

REACTOME = load_reactome_dict()

# ----------------------- Data helpers -----------------------
def _expr_path_for_study(study):
    # Return RNA-seq expression file path for a given study
    base = f"data/immuno/{study}"
    if study == "Kim":
        return os.path.join(base, "Kim_mRNA.norm3.txt")
    else:
        return os.path.join(base, f"{study}_mRNA.txt")

def _ssgsea_path_for_study(study):
    # Return ssGSEA NES matrix path
    return os.path.join(f"data/immuno/{study}", f"{study}_ssGSEA.txt")

def _biomarker_file_path(biomarker):
    # Path to network propagation output (.txt)
    return os.path.join(PROP_DIR, f"{biomarker}.txt")

def _load_universe(study):
    # Load expression matrix and define gene universe (uppercased)
    xp = _expr_path_for_study(study)
    df = pd.read_csv(xp, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.upper()
    return set(df.index), df

def _top_n_genes_from_propagation(biomarker, universe, top_n=200):
    # Retrieve Top-N propagated genes intersected with study universe
    p = _biomarker_file_path(biomarker)
    bdf = pd.read_csv(p, sep="\t")
    bdf["gene_id"] = bdf["gene_id"].astype(str).str.upper()
    bdf = bdf.dropna(subset=["gene_id"])
    bdf = bdf.sort_values("propagate_score", ascending=False)
    bdf = bdf.drop_duplicates(subset=["gene_id"], keep="first")
    genes = [g for g in bdf["gene_id"].tolist() if g in universe]
    return genes[:top_n]

def _weights_from_propagation(biomarker, universe, top_n=200, eps=1e-12):
    # Construct per-gene weights:
    #  - Top-N genes: scaled propagation scores in [eps, 1]
    #  - Others: baseline eps
    p = _biomarker_file_path(biomarker)
    bdf = pd.read_csv(p, sep="\t")
    bdf["gene_id"] = bdf["gene_id"].astype(str).str.upper()
    bdf = bdf.dropna(subset=["gene_id"])
    bdf = bdf[bdf["gene_id"].isin(universe)].copy()
    if bdf.empty:
        return {g: eps for g in universe}, set(), []
    bdf = bdf.sort_values("propagate_score", ascending=False)
    top_set = set(bdf["gene_id"].head(top_n).tolist())
    tmp = bdf.set_index("gene_id")["propagate_score"]
    s_top = tmp.loc[tmp.index.intersection(top_set)]
    if s_top.empty:
        return {g: eps for g in universe}, set(), []
    s_min, s_max = float(s_top.min()), float(s_top.max())
    if s_max > s_min:
        s_scaled = (s_top - s_min) / (s_max - s_min)
    else:
        s_scaled = pd.Series(1.0, index=s_top.index)
    s_scaled = s_scaled.clip(lower=eps, upper=1.0)
    w = {g: (float(s_scaled[g]) if g in s_scaled.index else eps) for g in universe}
    top_n_list = [g for g in bdf["gene_id"].tolist() if g in top_set][:top_n]
    return w, top_set, top_n_list

def _odds_from_weights(w_dict, pathway_genes, universe, min_odds=1e-6, max_odds=1e6):
    # Compute odds ratio for Wallenius test:
    # ω = mean(weight_in_pathway) / mean(weight_outside)
    U = list(universe)
    P = set([g for g in pathway_genes if g in universe])
    if len(P) == 0 or len(P) == len(U):
        return 1.0
    wP = np.array([w_dict[g] for g in U if g in P], dtype=float)
    wO = np.array([w_dict[g] for g in U if g not in P], dtype=float)
    muP = float(np.mean(wP)) if wP.size > 0 else 0.0
    muO = float(np.mean(wO)) if wO.size > 0 else 0.0
    if muP <= 0 or muO <= 0:
        return 1.0
    omega = muP / muO
    return float(np.clip(omega, min_odds, max_odds))

def _run_wallenius_ora(top_genes, universe, weights_dict, reactome_dict):
    # Weighted ORA (Wallenius) implementation:
    #   M  = |universe|
    #   m1 = |pathway ∩ universe|
    #   n  = |top_genes|
    #   k  = |pathway ∩ top_genes|
    #   ω  = mean(weight_in_pathway) / mean(weight_outside)
    #   p-value = P[X ≥ k] = sf(k-1; M, m1, n, ω)
    U  = set(universe)
    TG = set(top_genes)
    M, n = len(U), len(TG)
    if M == 0 or n == 0:
        return pd.DataFrame(columns=["Term", "pval", "qval", "overlap", "pw_count", "omega"])
    rows = []
    for term, genes in reactome_dict.items():
        P  = set(g.upper() for g in genes) & U
        m1 = len(P)
        if m1 == 0:
            continue
        k = len(P & TG)
        omega = _odds_from_weights(weights_dict, P, U)
        try:
            rv = nchypergeom_wallenius(M, m1, n, omega)
            p  = float(rv.sf(k-1)) if k > 0 else 1.0
            p  = max(min(p, 1.0), 0.0)
        except Exception:
            p = 1.0
        rows.append((term, p, k, m1, omega))
    if not rows:
        return pd.DataFrame(columns=["Term", "pval", "qval", "overlap", "pw_count", "omega"])
    df = pd.DataFrame(rows, columns=["Term", "pval", "overlap", "pw_count", "omega"])
    _, qvals, _, _ = multipletests(df["pval"].values, method="holm-sidak")
    df["qval"] = qvals
    df = df.sort_values(["qval", "pval"], ascending=[True, True])
    return df[["Term", "pval", "qval", "overlap", "pw_count", "omega"]]

# ----------------------- Main loop -----------------------
summary_rows = []
for STUDY in STUDIES:
    try:
        target = TARGET_BY_STUDY.get(STUDY)
        if target is None:
            print(f"[WARN] No target mapping for {STUDY}; skipping.")
            continue

        print(f"\n[{time.ctime()}] Study: {STUDY} | Target(biomarker): {target}")

        # 1) Define gene universe
        universe, _ = _load_universe(STUDY)
        print(f"  Universe size: {len(universe):,}")

        # 2) Retrieve Top-N genes and weights from propagation
        top_genes = _top_n_genes_from_propagation(target, universe, top_n=TOP_N)
        weights_dict, top_set, _ = _weights_from_propagation(target, universe, top_n=TOP_N, eps=1e-12)
        print(f"  Top genes (∩ universe): {len(top_genes)}  | weights on universe: {len(weights_dict)}")

        if len(top_genes) == 0:
            print(f"  [WARN] No Top-{TOP_N} genes intersect the universe for {STUDY}.")
            continue

        # 3) Weighted ORA using Wallenius distribution
        oraW_df = _run_wallenius_ora(top_genes, universe, weights_dict, REACTOME)
        if oraW_df.empty:
            print("  [WARN] Wallenius ORA produced no rows.")
            continue

        # 4) Select significant pathways by q-value threshold
        sigW_df = oraW_df[oraW_df["qval"] <= QVAL_CUTOFF].copy()
        sig_terms = sigW_df["Term"].tolist()
        print(f"  Wallenius-ORA significant (q ≤ {QVAL_CUTOFF}): {len(sig_terms)}")

        # 5) Filter ssGSEA NES matrix to significant pathways
        ssgsea_path = _ssgsea_path_for_study(STUDY)
        if not os.path.exists(ssgsea_path):
            raise FileNotFoundError(f"ssGSEA not found: {ssgsea_path}")
        nes = pd.read_csv(ssgsea_path, sep="\t", index_col=0)
        nes_filt = nes.loc[nes.index.intersection(sig_terms)].copy()
        print(f"  ssGSEA rows before: {nes.shape[0]} | after Wallenius filter: {nes_filt.shape[0]}")

        # --- Save outputs (parallel to NetBio original structure) ---
        out_dir = os.path.join(FO_ROOT_W, STUDY)
        os.makedirs(out_dir, exist_ok=True)

        oraW_path = os.path.join(out_dir, f"{STUDY}_{target}_Wallenius_ORA.tsv")
        oraW_df.to_csv(oraW_path, sep="\t", index=False)

        selW_path = os.path.join(out_dir, f"{STUDY}_{target}_Wallenius_selected_pathways.txt")
        pd.Series(sig_terms, name="selected_pathways").to_csv(selW_path, index=False, header=False)

        featW_path = os.path.join(out_dir, f"{STUDY}_ssGSEA_{target}_Wallenius_features.tsv")
        nes_filt.to_csv(featW_path, sep="\t")

        summary_rows.append({
            "Study": STUDY,
            "Target": target,
            "Universe": len(universe),
            "TopN_used": len(top_genes),
            "ORA_sig_terms": len(sig_terms),
            "Feature_rows": nes_filt.shape[0],
            "Feature_cols(samples)": nes_filt.shape[1],
            "ORA_table": oraW_path,
            "Selected_terms": selW_path,
            "Features_file": featW_path
        })

    except Exception as e:
        print(f"[ERROR] {STUDY}: {e}")

# ----------------------- Summary -----------------------
summary_wallenius = pd.DataFrame(summary_rows)
print("\n[SUMMARY] Wallenius-ORA outputs")
print(summary_wallenius)
