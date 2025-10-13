################################################################################
# Script: 0b_ssGSEA.py
# Author: Arianna Rigamonti
# Description: Compute single-sample Gene Set Enrichment Analysis (ssGSEA)
#              for all immunotherapy datasets using normalized expression data.
#              Reactome 2024 gene sets are used as the reference library.
#              For each dataset, the NES matrix (pathways × samples)
#              is exported for downstream enrichment filtering.
################################################################################

import os
import time
import pandas as pd
import gseapy as gp

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
os.chdir("path_to_your_repo")  # CHANGE to your local repo path

print(f"Working directory: {os.getcwd()}")

STUDIES = [
    "Auslander", "Gide", "Huang", "IMvigor210",
    "Kim", "Liu", "PEOPLE", "Prat", "Ravi", "Riaz"
]

GENESET_NAME = "Reactome_Pathways_2024"
ORGANISM = "Human"
THREADS = 10
SEED = 42

# ------------------------------------------------------------------------------
# Load gene sets
# ------------------------------------------------------------------------------
print(f"[{time.ctime()}] Loading gene sets: {GENESET_NAME}")
gene_sets = gp.get_library(name=GENESET_NAME, organism=ORGANISM)
print(f"  → {len(gene_sets):,} Reactome pathways loaded.")

# ------------------------------------------------------------------------------
# Run ssGSEA for each study
# ------------------------------------------------------------------------------
for STUDY in STUDIES:
    OUTDIR = f"data/immuno/{STUDY}"
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"\n[{time.ctime()}] Starting ssGSEA for {STUDY}")

    # --- Load normalized expression data ---
    try:
        if STUDY == "Kim":
            expr_path = f"{OUTDIR}/{STUDY}_mRNA.norm3.txt"
        else:
            expr_path = f"{OUTDIR}/{STUDY}_mRNA.txt"

        expr = pd.read_csv(expr_path, sep="\t", index_col=0)
        expr.index = expr.index.str.upper()
        print(f"  Expression matrix: {expr.shape[0]} genes × {expr.shape[1]} samples")
    except Exception as e:
        print(f"[ERROR] {STUDY}: failed to load expression file ({e})")
        continue

    # --- Run ssGSEA using Reactome 2024 ---
    try:
        ss = gp.ssgsea(
            data=expr,
            gene_sets=gene_sets,
            outdir=OUTDIR,
            sample_norm_method="rank",
            no_plot=True,
            threads=THREADS,
            seed=SEED,
            verbose=True
        )
    except Exception as e:
        print(f"[ERROR] {STUDY}: ssGSEA failed ({e})")
        continue

    # --- Extract NES matrix and export ---
    try:
        res = ss.res2d.copy()
        nes = res.pivot(index="Term", columns="Name", values="NES")
        out_file = os.path.join(OUTDIR, f"{STUDY}_ssGSEA.txt")
        nes.to_csv(out_file, sep="\t")
        print(f"[OK] {STUDY}: NES matrix saved → {out_file}")
        print(f"      {nes.shape[0]} pathways × {nes.shape[1]} samples")
    except Exception as e:
        print(f"[ERROR] {STUDY}: failed to save ssGSEA output ({e})")
        continue

print(f"\n[{time.ctime()}] ssGSEA completed for all datasets.")
