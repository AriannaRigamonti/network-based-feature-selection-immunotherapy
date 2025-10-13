################################################################################
# Script: 1_network_propagation.py
# Author: Arianna Rigamonti
# Description: Build a STRING PPI graph (score ≥700), restrict to the largest
#              connected component (LCC), map ENSP IDs to a single canonical
#              HGNC symbol, and run PageRank-based propagation for each
#              immunotherapy biomarker defined in Marker_summary.txt.
# Output: For each biomarker, a TSV with columns:
#         gene_id (HGNC), string_protein_id (ENSP), propagate_score.      
################################################################################

import os
os.chdir("path_to_your_repo")  # CHANGE to your local repo path
print(os.getcwd())

# ---- Imports (standard + analysis/plotting + graph) ----
import os
import glob
import time
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch figure/file generation
import matplotlib.pyplot as plt
import networkx as nx

# ---- Environment versions (useful for provenance and reproducibility) ----
print(f"Pandas version: {pd.__version__}")
print(f"NetworkX version: {nx.__version__}")

# ---- Scope ----
# Network propagation for immunotherapy biomarkers:
# - Build STRING PPI (filtered by combined score ≥700).
# - Map proteins to a single canonical HGNC symbol per ENSP.
# - Run PageRank-based propagation on the LCC.
# - Write one TSV per biomarker with columns:
#   gene_id (HGNC), string_protein_id (ENSP), propagate_score.

# ---- Reproducibility and I/O setup ----
VERSION = "v12" # STRING database version: "v11" or "v12"
INTERACTION = "full" # "full" or "physical"
# Note: "physical" is a subset of "full" (functional + physical associations).
STRING_CUTOFF = 700
FO_DIR = f"results/0_data_collection_and_preprocessing/{VERSION}/{INTERACTION}"
os.makedirs(FO_DIR, exist_ok=True)

print(f"[{time.ctime()}] Start")

# ---- Load immunotherapy biomarkers (HGNC symbols expected) ----
ib = pd.read_csv("data/Marker_summary.txt", sep="\t")
assert not ib.empty, "Marker_summary.txt is empty or not found."

# ---- Load STRING alias mapping (broad HGNC→ENSP seeding support) ----
# Use curated alias sources to robustly map seed symbols to ENSP within LCC.
alias_cols = ["string_protein_id", "alias", "source"]
aliases = pd.read_csv(
    f"data/STRING/9606.protein.aliases.{VERSION}.0.txt",
    sep="\t",
    comment="#",
    header=None,
    names=alias_cols,
    dtype={"string_protein_id": str, "alias": str, "source": str}
)
print(f"Aliases loaded: {aliases.shape[0]:,} rows")

preferred_sources = ["Gene_Name", "Ensembl_HGNC"]
aliases_pref = aliases[aliases["source"].isin(preferred_sources)].copy()
if aliases_pref.empty:
    aliases_pref = aliases[aliases["source"].str.contains("HGNC|Gene", na=False)].copy()
print(f"Preferred aliases subset: {aliases_pref.shape[0]:,} rows")

# ---- Load STRING links and build an undirected weighted graph ----
# Select "full" (functional+physical) or "physical" interactions based on INTERACTION.
if INTERACTION == "full":
    links = pd.read_csv(
        f"data/STRING/9606.protein.links.{VERSION}.0.txt",
        sep=r"\s+",
        engine="python"
    )
elif INTERACTION == "physical":
    links = pd.read_csv(
        f"data/STRING/9606.protein.physical.links.{VERSION}.0.txt",
        sep=r"\s+",
        engine="python"
    )
expected_cols = {"protein1", "protein2", "combined_score"}
assert expected_cols.issubset(links.columns), f"Missing columns in links: {links.columns.tolist()}"

# Restrict to human proteins and apply score threshold
links = links[links["protein1"].str.startswith("9606.") & links["protein2"].str.startswith("9606.")]
links = links[links["combined_score"] >= STRING_CUTOFF].copy()
print(f"Links after cutoff (≥{STRING_CUTOFF}): {links.shape[0]:,}")

# Build graph with combined_score as edge weight
G_all = nx.from_pandas_edgelist(
    links,
    source="protein1",
    target="protein2",
    edge_attr="combined_score",
    create_using=nx.Graph()
)

print(f"Graph (thresholded) - nodes: {G_all.number_of_nodes():,}, edges: {G_all.number_of_edges():,}")

# ---- Largest Connected Component (LCC) extraction ----
if G_all.number_of_nodes() == 0:
    raise ValueError("Graph is empty after applying the score cutoff. Lower the cutoff or check inputs.")

largest_cc_nodes = max(nx.connected_components(G_all), key=len)
G = G_all.subgraph(largest_cc_nodes).copy()
print(f"LCC - nodes: {G.number_of_nodes():,}, edges: {G.number_of_edges():,}")

# ---- Canonical ENSP→HGNC mapping via STRING protein.info ----
# Ensure a single preferred_name (HGNC symbol) per ENSP to avoid alias proliferation.
info = pd.read_csv(f"data/STRING/9606.protein.info.{VERSION}.0.txt", sep="\t")
# Typical columns:
#  - v11: 'protein_external_id' (e.g., 9606.ENSP...), 'preferred_name'
#  - v12: '#string_protein_id', 'preferred_name'
if VERSION == "v11":
    ensp2symbol = dict(zip(info["protein_external_id"], info["preferred_name"]))
elif VERSION == "v12":
    ensp2symbol = dict(zip(info["#string_protein_id"], info["preferred_name"]))

# ---- Build annotation dictionary restricted to LCC: ENSP → single HGNC symbol ----
anno_dic = {}
for ensp in G.nodes():
    sym = ensp2symbol.get(ensp, None)
    if isinstance(sym, str) and len(sym) > 0:
        anno_dic[ensp] = sym
print(f"Annotated {len(anno_dic):,} / {G.number_of_nodes():,} LCC nodes with preferred HGNC symbol.")

print("PPI network preparation completed.")
print(f"[{time.ctime()}] End")

# --- Network propagation over the LCC (PageRank with personalization) ---
# For each biomarker (seed set), run PageRank on the LCC and export results.
print(f"[{time.ctime()}] Run network propagation")

# ---- Safety checks for required objects ----
assert "G" in globals() and isinstance(G, nx.Graph), "Graph G not found."
assert "anno_dic" in globals() and isinstance(anno_dic, dict), "anno_dic not found."
assert "aliases_pref" in globals() and isinstance(aliases_pref, pd.DataFrame), "aliases_pref not found."
assert "ib" in globals() and isinstance(ib, pd.DataFrame), "Marker_summary (ib) not found."

# ---- Build alias→ENSP map limited to LCC (robust HGNC seeding) ----
lcc_nodes = set(G.nodes())
alias_to_ensp = (
    aliases_pref.loc[aliases_pref["string_protein_id"].isin(lcc_nodes), ["alias", "string_protein_id"]]
    .dropna()
    .groupby("alias")["string_protein_id"].apply(set)
    .to_dict()
)

# ---- Validate Marker_summary schema ----
required_cols = {"Name", "Feature", "Gene_list"}
missing = required_cols.difference(ib.columns)
if missing:
    raise ValueError(f"Marker_summary is missing required columns: {sorted(missing)}")

# ---- Helper for safe file naming ----
def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(s))

# ---- Select biomarkers of interest (features labeled as targets) ----
subset = ib[ib["Feature"].astype(str).str.contains("target", case=False, na=False)].copy()

# ---- PageRank propagation per biomarker ----
start_np = time.ctime()
for _, row in subset.iterrows():
    biomarker = str(row["Name"])
    feature = str(row["Feature"])
    gene_list_raw = str(row["Gene_list"])

    out_fname = f"{_safe_name(biomarker)}.txt"
    out_path = os.path.join(FO_DIR, out_fname)

    # Skip recomputation if output is already present
    if os.path.exists(out_path):
        continue

    print(f"\t[seed: {biomarker}] {time.ctime()}")

    # Parse seed symbols (colon-separated HGNC symbols expected)
    biomarker_genes = [g.strip() for g in gene_list_raw.split(":") if g.strip()]
    if not biomarker_genes:
        print(f"\t\tNo genes parsed for {biomarker}; skipping.")
        continue

    # Map seed HGNC symbols to LCC ENSPs
    seed_ensps = set()
    for gsym in biomarker_genes:
        if gsym in alias_to_ensp:
            seed_ensps.update(alias_to_ensp[gsym])

    if not seed_ensps:
        print(f"\t\tNo LCC nodes found for seeds of {biomarker}; skipping.")
        continue

    # Personalization vector: seeds = 1.0, others = 0.0
    personalization = {n: 1.0 if n in seed_ensps else 0.0 for n in lcc_nodes}

    # Weighted PageRank (combined_score as edge weight)
    propagate_scores = nx.pagerank(
        G,
        alpha=0.85,
        personalization=personalization,
        weight="combined_score",
        max_iter=100,
        tol=1e-6,
    )

    # Assemble output using a single canonical HGNC symbol per ENSP
    records = []
    for ensp, score in propagate_scores.items():
        sym = anno_dic.get(ensp, "NA")
        records.append(
            {
                "gene_id": sym,                   # canonical HGNC symbol
                "string_protein_id": ensp,
                "propagate_score": score,
            }
        )

    output_df = pd.DataFrame.from_records(
        records, columns=["gene_id", "string_protein_id", "propagate_score"]
    )

    # Keep only mapped symbols; drop duplicates keeping the highest score
    output_df = output_df[output_df["gene_id"] != "NA"].copy()
    output_df = output_df.sort_values(by=["propagate_score"], ascending=False)
    output_df = output_df.drop_duplicates(subset=["gene_id"], keep="first")

    # Stable sort for readability (desc by score, asc by gene_id)
    output_df = output_df.sort_values(
        by=["propagate_score", "gene_id"], ascending=[False, True]
    ).reset_index(drop=True)

    # Write biomarker-specific propagation scores
    output_df.to_csv(out_path, sep="\t", index=False)

end_np = time.ctime()
print(f"Process complete // start: {start_np}, end: {end_np}")
