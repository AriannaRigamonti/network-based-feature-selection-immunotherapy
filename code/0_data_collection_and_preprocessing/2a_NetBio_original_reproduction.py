################################################################################
# Script: 2a_NetBio_original_reproduction.py
# Author: Arianna Rigamonti
# Description: Reproduce the original NetBio pipeline:
#   - load study-specific expression universes;
#   - read biomarker-specific propagation results (Top-N genes);
#   - run hypergeometric ORA against Reactome (2024) with multiple-testing control;
#   - filter ssGSEA NES matrices by ORA-significant terms;
#   - export per-study feature matrices and logs.
################################################################################

import os
os.chdir("path_to_your_repo")  # CHANGE to your local repo path
print(os.getcwd())
import os, time
import pandas as pd
import numpy as np
import gseapy as gp
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests

# ----------------------- Configuration -----------------------
# Studies and targets (per original NetBio mapping)
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

# Inherit VERSION/INTERACTION if already defined in the session; else set defaults
VERSION = globals().get("VERSION", "v12")          # "v11" or "v12"
INTERACTION = globals().get("INTERACTION", "full") # "full" or "physical"
PROP_DIR = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"

# ORA parameters (per NetBio: fixed Top-N, FDR threshold)
TOP_N = 200          # number of top genes by propagation score
QVAL_CUTOFF = 0.01   

# Output root for this export
FO_ROOT = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}/NetBio_original"
os.makedirs(FO_ROOT, exist_ok=True)

# ----------------------- Reactome pathways -----------------------
def load_reactome():
    # Fetch Reactome 2024 from gseapy; return dict term -> list of uppercased symbols
    lib = gp.get_library(name="Reactome_Pathways_2024", organism="Human")
    if isinstance(lib, dict):
        reactome = {term: [str(g).upper() for g in genes]
                    for term, genes in lib.items()}
    else:
        # Fallback: request DataFrame and construct dictionary
        lib_df = gp.get_library(name="Reactome_Pathways_2024",
                                organism="Human", return_dataframe=True)
        cols_lower = {c.lower(): c for c in lib_df.columns}
        term_col = (cols_lower.get("term") or cols_lower.get("name")
                    or list(lib_df.columns)[0])
        gene_col = (cols_lower.get("gene_symbol") or cols_lower.get("genes")
                    or list(lib_df.columns)[1])
        reactome = (lib_df
                    .rename(columns={term_col: "Term", gene_col: "Gene"})
                    .groupby("Term")["Gene"]
                    .apply(lambda s: [str(x).upper() for x in s.tolist()])
                    .to_dict())
    # Cleanup: drop empty sets, deduplicate symbols
    reactome = {k: sorted(set(v)) for k, v in reactome.items() if len(v) > 0}
    print(f"[INFO] Loaded Reactome: {len(reactome):,} terms.")
    if not reactome:
        raise RuntimeError("Reactome dictionary is empty — check gseapy/get_library.")
    return reactome

REACTOME = load_reactome()

# ----------------------- Helpers -----------------------
def _expr_path_for_study(study):
    # Build path to study-level expression matrix
    base = f"data/immuno/{study}"
    if study == "Kim":
        return os.path.join(base, "Kim_mRNA.norm3.txt")
    else:
        return os.path.join(base, f"{study}_mRNA.txt")

def _ssgsea_path_for_study(study):
    # Path to ssGSEA NES matrix (rows=terms, cols=samples)
    return os.path.join(f"data/immuno/{study}", f"{study}_ssGSEA.txt")

def _biomarker_file_path(biomarker):
    # Path to network propagation output (TSV with gene_id, string_protein_id, propagate_score)
    return os.path.join(PROP_DIR, f"{biomarker}.txt")

def _load_universe(study):
    # Load expression matrix and define the universe as all measured genes (uppercased)
    xp = _expr_path_for_study(study)
    df = pd.read_csv(xp, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.upper()
    return set(df.index), df

def _top_n_genes_from_propagation(biomarker, universe, top_n=200):
    # Read propagation scores and return Top-N genes intersected with the expression universe
    p = _biomarker_file_path(biomarker)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing propagation file: {p}")
    bdf = pd.read_csv(p, sep="\t")  # expects ['gene_id','string_protein_id','propagate_score']
    if "gene_id" not in bdf.columns or "propagate_score" not in bdf.columns:
        raise ValueError(f"{os.path.basename(p)} missing required columns.")
    bdf["gene_id"] = bdf["gene_id"].astype(str).str.upper()
    bdf = bdf.dropna(subset=["gene_id"])
    bdf = bdf.sort_values("propagate_score", ascending=False)
    bdf = bdf.drop_duplicates(subset=["gene_id"], keep="first")
    genes = [g for g in bdf["gene_id"].tolist() if g in universe]
    return genes[:top_n]

def _run_hypergeom_ora(top_genes, universe, reactome_dict):
    # Hypergeometric ORA (population=universe; draws=Top-N; successes=pathway genes)
    M = len(universe)                 # population size
    N = len(set(top_genes))           # number of draws
    if M == 0 or N == 0:
        raise RuntimeError("Empty universe or no top genes.")
    rows = []
    U = set(universe)
    TG = set(top_genes)
    for term, genes in reactome_dict.items():
        pw_genes = set(g.upper() for g in genes) & U
        n = len(pw_genes)             # number of success states in population
        if n == 0:
            continue
        k = len(pw_genes & TG)        # observed successes
        p = float(stat.hypergeom.sf(k - 1, M, n, N))  # right-tail p-value
        rows.append((term, p, k, n))
    if not rows:
        return pd.DataFrame(columns=["Term", "pval", "qval", "overlap", "pw_count"])
    df = pd.DataFrame(rows, columns=["Term", "pval", "overlap", "pw_count"])
    # Multiple-testing correction (Holm–Sidak via statsmodels default)
    _, qvals, _, _ = multipletests(df["pval"].values)
    df["qval"] = qvals
    df = df.sort_values(["qval", "pval"], ascending=[True, True])
    return df[["Term", "pval", "qval", "overlap", "pw_count"]]

# ----------------------- Main loop -----------------------
summary_rows = []
for STUDY in STUDIES:
    try:
        target = TARGET_BY_STUDY.get(STUDY)
        if target is None:
            print(f"[WARN] No target mapping for {STUDY}; skipping.")
            continue

        print(f"\n[{time.ctime()}] Study: {STUDY} | Target: {target}")

        # 1) Define universe (all measured genes for the study)
        universe, expr_df = _load_universe(STUDY)
        print(f"  Universe size: {len(universe):,}")

        # 2) Select Top-N propagated genes within the universe (NetBio step)
        top_genes = _top_n_genes_from_propagation(target, universe, top_n=TOP_N)
        print(f"  Top genes (after universe ∩): {len(top_genes)}")

        # 3) Run hypergeometric ORA with multiple-testing control
        ora_df = _run_hypergeom_ora(top_genes, universe, REACTOME)
        if ora_df.empty:
            print("  [WARN] ORA produced no rows.")
            continue

        # 4) Apply NetBio threshold and collect significant pathways
        sig_df = ora_df[ora_df["qval"] <= QVAL_CUTOFF].copy()
        sig_terms = sig_df["Term"].tolist()
        print(f"  ORA-significant pathways (q ≤ {QVAL_CUTOFF}): {len(sig_terms)}")

        # 5) Filter ssGSEA NES matrix to ORA-significant terms
        ssgsea_path = _ssgsea_path_for_study(STUDY)
        if not os.path.exists(ssgsea_path):
            raise FileNotFoundError(f"ssGSEA matrix not found: {ssgsea_path}")
        nes = pd.read_csv(ssgsea_path, sep="\t", index_col=0)   # rows=Term, cols=samples
        nes_filt = nes.loc[nes.index.intersection(sig_terms)].copy()
        print(f"  ssGSEA rows before: {nes.shape[0]} | after ORA filter: {nes_filt.shape[0]}")

        # --- Save outputs ---
        out_dir = os.path.join(FO_ROOT, STUDY)
        os.makedirs(out_dir, exist_ok=True)

        ora_path = os.path.join(out_dir, f"{STUDY}_{target}_NetBio_ORA.tsv")
        ora_df.to_csv(ora_path, sep="\t", index=False)

        sel_path = os.path.join(out_dir, f"{STUDY}_{target}_NetBio_selected_pathways.txt")
        pd.Series(sig_terms, name="selected_pathways").to_csv(sel_path, index=False, header=False)

        feat_path = os.path.join(out_dir, f"{STUDY}_ssGSEA_{target}_NetBio_features.tsv")
        nes_filt.to_csv(feat_path, sep="\t")

        summary_rows.append({
            "Study": STUDY,
            "Target": target,
            "Universe": len(universe),
            "TopN_used": len(top_genes),
            "ORA_sig_terms": len(sig_terms),
            "Feature_rows": nes_filt.shape[0],
            "Feature_cols(samples)": nes_filt.shape[1],
            "ORA_table": ora_path,
            "Selected_terms": sel_path,
            "Features_file": feat_path
        })
    except Exception as e:
        print(f"[ERROR] {STUDY}: {e}")

# ----------------------- Summary table -----------------------
summary_df = pd.DataFrame(summary_rows)
print(summary_df)
