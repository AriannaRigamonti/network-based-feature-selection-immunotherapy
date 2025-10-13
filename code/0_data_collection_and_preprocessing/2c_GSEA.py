################################################################################
# Script: 2c_GSEA.py
# Author: Arianna Rigamonti
# Description: Perform Gene Set Enrichment Analysis (GSEA) using propagation-
#              derived rankings and Reactome 2024 gene sets. 
#              The script builds .rnk files, runs GSEApy prerank for each dataset,
#              generates visual outputs (dotplots, enrichment maps), and 
#              filters results based on FDR and NES percentile thresholds.
################################################################################

import os
os.chdir("path_to_your_repo")  # CHANGE to your local repo path
print(os.getcwd())

# ------------------------------------------------------------------------------
# Library import and environment setup
# ------------------------------------------------------------------------------
import os
import glob
import time
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch saving
import matplotlib.pyplot as plt

import gseapy as gp
from gseapy import gseaplot, gseaplot2, dotplot, enrichment_map
import networkx as nx

# Version check
print(f"Pandas version: {pd.__version__}")
print(f"NetworkX version: {nx.__version__}")
print(f"GSEApy version: {gp.__version__}")

# ------------------------------------------------------------------------------
# Step 1 – Build .rnk files from propagation outputs
# ------------------------------------------------------------------------------
VERSION = "12"
INTERACTION = "full"
FO_DIR = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"
RNK_DIR = os.path.join(FO_DIR, "rnk")
os.makedirs(RNK_DIR, exist_ok=True)

# Create .rnk files (gene_id, propagate_score) from propagation outputs
tsv_files = [f for f in os.listdir(FO_DIR) if f.endswith(".txt")]

for fname in tsv_files:
    in_path = os.path.join(FO_DIR, fname)
    try:
        df = pd.read_csv(in_path, sep="\t")
    except Exception as e:
        print(f"[WARN] Skipping {fname}: {e}")
        continue

    if not {"gene_id", "propagate_score"}.issubset(df.columns):
        print(f"[WARN] {fname} missing required columns; skipping.")
        continue

    rnk = (
        df[["gene_id", "propagate_score"]]
        .dropna()
        .groupby("gene_id", as_index=False)["propagate_score"].max()
        .sort_values("propagate_score", ascending=False)
    )

    base = os.path.splitext(fname)[0]
    out_path = os.path.join(RNK_DIR, f"{base}.rnk")
    rnk.to_csv(out_path, sep="\t", index=False, header=False)
    print(f"[OK] Wrote {out_path}")

print("All .rnk files generated.")

# ------------------------------------------------------------------------------
# Step 2 – GSEA prerank execution for each study
# ------------------------------------------------------------------------------
names = gp.get_library_name()
print("Reactome_Pathways_2024" in names)

reactome_pathways = gp.get_library(name="Reactome_Pathways_2024")
print("Number of pathways in Reactome 2024:", len(reactome_pathways))

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gseapy as gp
from gseapy import dotplot, enrichment_map
import networkx as nx

# Configuration
STUDIES = ["Auslander", "Gide", "Huang", "IMvigor210", "Kim", "Liu", "PEOPLE", "Prat", "Ravi", "Riaz"]
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

VERSION = "v12"
INTERACTION = "full"
PERMUTATIONS = 1000
N_GENES = None
np.random.seed(42)
threads = min(10, os.cpu_count() or 1)

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def break_prerank_ties(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Add small epsilon within ties to enforce strict descending rank order."""
    score = df.iloc[:, 0].astype(float)
    tmp = pd.DataFrame({"score": score}, index=df.index)
    tmp["gene"] = tmp.index.astype(str)
    tmp = tmp.sort_values(["score", "gene"], ascending=[False, True])
    tmp["_within_tie"] = tmp.groupby("score", sort=False).cumcount()
    tmp["score_adj"] = tmp["score"] + tmp["_within_tie"] * eps
    df.iloc[:, 0] = tmp["score_adj"].reindex(df.index)
    return df

def load_gene_sets_as_dict(name="Reactome_Pathways_2024", organism="Human"):
    """Load gene sets from GSEApy as dict {term: [genes]} with uppercased symbols."""
    lib = gp.get_library(name=name, organism=organism)
    if isinstance(lib, dict):
        return {term: [str(g).upper() for g in genes] for term, genes in lib.items()}
    lib_df = gp.get_library(name=name, organism=organism, return_dataframe=True)
    cols_lower = {c.lower(): c for c in lib_df.columns}
    term_col = cols_lower.get("term") or cols_lower.get("name") or cols_lower.get("pathway") or list(lib_df.columns)[0]
    gene_col = cols_lower.get("gene_symbol") or cols_lower.get("genes") or cols_lower.get("gene") or list(lib_df.columns)[1]
    gs_dict = (
        lib_df.rename(columns={term_col: "Term", gene_col: "Gene"})
        .groupby("Term")["Gene"]
        .apply(lambda s: [str(x).upper() for x in s.tolist()])
        .to_dict()
    )
    return gs_dict

def trim_gene_sets_to_universe(gene_sets_dict, universe, min_size=15, max_size=500):
    """Intersect each gene set with dataset universe and retain within size limits."""
    trimmed = {}
    for term, genes in gene_sets_dict.items():
        g = list(set(g.upper() for g in genes) & universe)
        if min_size <= len(g) <= max_size:
            trimmed[term] = g
    return trimmed

# ------------------------------------------------------------------------------
# GSEA prerank loop
# ------------------------------------------------------------------------------
FO_DIR = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"
RNK_DIR = os.path.join(FO_DIR, "rnk")
GSEAPY_DIR = os.path.join(FO_DIR, f"gseapy95_{N_GENES}")
os.makedirs(GSEAPY_DIR, exist_ok=True)

if not os.path.isdir(RNK_DIR):
    raise FileNotFoundError(f"RNK_DIR not found: {RNK_DIR}")

for STUDY in STUDIES:
    target_sig = TARGET_BY_STUDY.get(STUDY)
    if target_sig is None:
        print(f"[WARN] [{STUDY}] No target mapping found. Skipping.")
        continue

    OUTDIR = f"data/immuno/{STUDY}"

    # Assign GSEA weighting and FDR cutoff
    if STUDY in ("Huang", "Prat", "PEOPLE"):
        WEIGHT = 0
        CUTOFF = 0.25
    else:
        WEIGHT = 1
        CUTOFF = 0.25

    # Load expression matrix to define gene universe
    try:
        expr_path = f"{OUTDIR}/{STUDY}_mRNA.norm3.txt" if STUDY == "Kim" else f"{OUTDIR}/{STUDY}_mRNA.txt"
        expr = pd.read_csv(expr_path, index_col=0, sep="\t")
    except Exception as e:
        print(f"[WARN] [{STUDY}] Failed to read expression file: {e}. Skipping.")
        continue

    expr.index = expr.index.str.upper()
    universe = set(expr.index)

    # Load Reactome pathways and trim
    print(f"[{time.ctime()}] [{STUDY}] Load and trim Reactome pathways")
    try:
        gs_dict = load_gene_sets_as_dict(name="Reactome_Pathways_2024", organism="Human")
        gene_sets = trim_gene_sets_to_universe(gs_dict, universe)
        if not gene_sets:
            print(f"[WARN] [{STUDY}] No valid gene sets after trimming. Skipping.")
            continue
    except Exception as e:
        print(f"[WARN] [{STUDY}] Gene set load/trim failed: {e}. Skipping.")
        continue

    # Load .rnk file for the target signature
    rnk_file = f"{target_sig}.rnk"
    path = os.path.join(RNK_DIR, rnk_file)
    try:
        df = pd.read_csv(path, header=None, index_col=0, sep="\t")
    except Exception as e:
        print(f"[WARN] [{STUDY}] RNK file '{rnk_file}' missing or unreadable: {e}. Skipping.")
        continue

    df.index = df.index.astype(str).str.upper()
    df = df[df.index.isin(universe)].copy()
    if df.empty:
        print(f"[WARN] [{STUDY}] No overlapping genes between RNK and expression. Skipping.")
        continue

    df = df.sort_values(df.columns[0], ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    df = break_prerank_ties(df)
    if N_GENES is not None:
        df = df.iloc[:N_GENES].copy()

    if df.shape[0] < 15:
        print(f"[WARN] [{STUDY}] Only {df.shape[0]} genes after filtering; skipping.")
        continue

    # Run GSEA prerank
    print(f"[{time.ctime()}] [{STUDY}] Run GSEA prerank for target: {target_sig}")
    out_dir = os.path.join(GSEAPY_DIR, STUDY, target_sig)
    os.makedirs(out_dir, exist_ok=True)

    try:
        pre_res = gp.prerank(
            rnk=df,
            gene_sets=gene_sets,
            threads=threads,
            permutation_num=PERMUTATIONS,
            weight=WEIGHT,
            outdir=out_dir,
            format="png",
            seed=42,
            verbose=True,
        )

        res2d = getattr(pre_res, "res2d", None)
        if res2d is None or res2d.empty:
            print(f"[{time.ctime()}] [{STUDY}] No enriched pathways.")
            continue

        res2d = res2d[res2d["NES"] > 0].copy()
        if res2d.empty:
            print(f"[{time.ctime()}] [{STUDY}] No positive NES terms.")
            continue
        res2d = res2d.sort_values("FDR q-val", ascending=True)

        # Dotplot visualization
        _ = dotplot(
            res2d,
            column="FDR q-val",
            title=f"Reactome (top by FDR) — {target_sig}",
            size=6,
            figsize=(6, 8),
            cutoff=CUTOFF,
            show_ring=False,
            ofname=os.path.join(out_dir, "dotplot.png"),
        )
        print(f"[{time.ctime()}] [{STUDY}] Dotplot saved")

        # Enrichment map and network export
        nodes, edges = enrichment_map(res2d, cutoff=CUTOFF, top_term=20)

        if "id" not in nodes.columns:
            nodes = nodes.reset_index()
            if "id" not in nodes.columns:
                nodes = nodes.rename(columns={nodes.columns[0]: "id"})
        if "Term" not in nodes.columns:
            for alt in ["term", "name", "pathway"]:
                if alt in nodes.columns:
                    nodes = nodes.rename(columns={alt: "Term"})
                    break

        def _pick(edges_df, candidates):
            for c in candidates:
                if c in edges_df.columns:
                    return c
            return None

        src_col = _pick(edges, ["src_idx", "src", "source", "from"])
        tgt_col = _pick(edges, ["targ_idx", "targ", "target", "to"])
        if src_col is None or tgt_col is None:
            raise ValueError(f"Edge columns not found: {edges.columns.tolist()}")

        nodes.to_csv(os.path.join(out_dir, "enrichmap_nodes.csv"), index=False)
        edges.to_csv(os.path.join(out_dir, "enrichmap_edges.csv"), index=False)

        G = nx.from_pandas_edgelist(
            edges,
            source=src_col,
            target=tgt_col,
            edge_attr=[c for c in ["jaccard_coef", "overlap_coef", "overlap_genes"] if c in edges.columns],
        )
        node_attr = nodes.set_index("id").to_dict(orient="index")
        nx.set_node_attributes(G, node_attr)

        from adjustText import adjust_text
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=42)
            sizes, colors, labels = [], [], {}
            for n in G.nodes():
                nd = node_attr.get(n, {})
                sizes.append(float(nd.get("Hits_ratio", 0.2)) * 1000.0)
                colors.append(float(nd.get("NES", 0.0)))
                labels[n] = nd.get("Term", str(n))
            edge_w = [max(0.5, float(w) * 10.0) for _, _, w in G.edges(data="jaccard_coef", default=0.0)]

            fig, ax = plt.subplots(figsize=(20, 15))
            sc = nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=sizes, cmap=plt.cm.RdYlBu, ax=ax)
            nx.draw_networkx_edges(G, pos=pos, width=edge_w, edge_color="#CDDBD4", ax=ax)

            texts = []
            for n, (x, y) in pos.items():
                lbl = labels[n]
                texts.append(ax.text(x, y, lbl, fontsize=6, ha="center", va="center"))
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("NES")
            ax.set_axis_off()
            fig.savefig(os.path.join(out_dir, "enrichmap_network.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"[{time.ctime()}] [{STUDY}] Enrichment map network saved")
        print(f"[{time.ctime()}] [{STUDY}] GSEA prerank completed for {target_sig}")

    except Exception as e:
        print(f"[WARN] [{STUDY}] GSEA prerank failed for {target_sig}: {e}")

# ------------------------------------------------------------------------------
# Step 3 – Aggregate and filter results (percentile-based NES selection)
# ------------------------------------------------------------------------------
import os
import glob
import time
import numpy as np
import pandas as pd

STUDIES = ["Auslander", "Gide", "Huang", "IMvigor210", "Kim", "Liu", "PEOPLE", "Prat", "Ravi", "Riaz"]
VERSION = globals().get("VERSION", "v12")
INTERACTION = globals().get("INTERACTION", "full")
N_GENES = globals().get("N_GENES", None)
FO_DIR = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"
GSEAPY_DIR = os.path.join(FO_DIR, f"gseapy95_{N_GENES}")

# Helper functions for post-processing
def _normalize_columns(df):
    colmap = {c.lower().strip(): c for c in df.columns}
    lookup = {}
    for cand in ["term", "name", "pathway"]:
        if cand in colmap:
            lookup["Term"] = colmap[cand]; break
    if "es" in colmap:  lookup["ES"] = colmap["es"]
    if "nes" in colmap: lookup["NES"] = colmap["nes"]
    for cand in ["nom p-val", "p-value", "pvalue", "nominal p-val"]:
        if cand in colmap: lookup["NOM p-val"] = colmap[cand]; break
    for cand in ["fdr q-val", "fdr", "fdr qvalue"]:
        if cand in colmap: lookup["FDR q-val"] = colmap[cand]; break
    for cand in ["fwer p-val", "fwer"]:
        if cand in colmap: lookup["FWER p-val"] = colmap[cand]; break
    df2 = df.copy()
    for std, real in lookup.items():
        if std != real:
            df2 = df2.rename(columns={real: std})
    return df2

def _coerce_float(series):
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series

def _subset_for_percentile(df_sig, direction):
    nes = df_sig["NES"].dropna()
    if direction == "pos":
        nes = nes[nes > 0]
    elif direction == "neg":
        nes = nes[nes < 0]
    return nes

def _compute_signature_percentiles(df_all, direction, perc, min_values=20):
    sig2perc = {}
    global_base = _subset_for_percentile(df_all, direction)
    global_thr = (np.nanpercentile(global_base, perc) if len(global_base) >= min_values else None)
    for sig, df_sig in df_all.groupby("Signature"):
        base = _subset_for_percentile(df_sig, direction)
        if len(base) >= min_values:
            sig2perc[sig] = np.nanpercentile(base, perc)
        else:
            sig2perc[sig] = global_thr
    return sig2perc

# Process results per study
for STUDY in STUDIES:
    FDR_CUTOFF = 0.25
    NES_DIRECTION = "pos"
    NES_PERC_VALUE = 95
    BASE_DIR = GSEAPY_DIR
    study_dir = os.path.join(BASE_DIR, STUDY)
    if not os.path.isdir(study_dir):
        print(f"[WARN] [{STUDY}] prerank output not found: {study_dir}. Skipping.")
        continue

    pattern = os.path.join(study_dir, "*", "gseapy.gene_set.prerank.report.csv")
    report_files = sorted(glob.glob(pattern))
    if not report_files:
        print(f"[WARN] [{STUDY}] No GSEApy reports found.")
        continue

    all_rows = []
    for path in report_files:
        signature = os.path.basename(os.path.dirname(path))
        try:
            df = pd.read_csv(path)
            df = _normalize_columns(df)
            keep_cols = [c for c in ["Term", "ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"] if c in df.columns]
            df = df[keep_cols].copy()
            for c in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
                if c in df.columns:
                    df[c] = _coerce_float(df[c])
            df.insert(0, "Signature", signature)
            all_rows.append(df)
        except Exception as e:
            print(f"[WARN] [{STUDY}] Skip {path}: {e}")

    if not all_rows:
        print(f"[WARN] [{STUDY}] No valid GSEA results.")
        continue

    res_all = pd.concat(all_rows, axis=0, ignore_index=True)
    sig2perc = _compute_signature_percentiles(res_all, NES_DIRECTION, NES_PERC_VALUE)

    mask = (res_all["FDR q-val"] <= FDR_CUTOFF) & (res_all["NES"] > 0)
    res_filt = res_all.loc[mask].copy()

    def _pass_nes_perc(row):
        thr = sig2perc.get(row["Signature"], None)
        if thr is None:
            return True
        return row["NES"] >= thr

    res_filt = res_filt[res_filt.apply(_pass_nes_perc, axis=1)]
    res_filt = res_filt.sort_values(["FDR q-val", "NES"], ascending=[True, False])

    os.makedirs(os.path.join(BASE_DIR, STUDY), exist_ok=True)
    out_table = os.path.join(BASE_DIR, STUDY, f"{STUDY}_summary_FDRle0p25_NESp95.tsv")
    res_filt.to_csv(out_table, sep="\t", index=False)
    print(f"[OK] [{STUDY}] Saved filtered summary: {out_table}")
