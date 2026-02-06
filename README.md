# Weighted pathway enrichment strategies for interpretable network-based biomarker discovery in immunotherapy response prediction

Authors: Arianna Rigamonti1, Luca Mauro Invernizzi2, Ghazal Farhikhteh2, Vanja Mišković1, Francesco Trovò1 and Arsela Prelaj2. 

1. Department of Electronic, Information and Bioengineering, Politecnico di Milano, Milan, Italy
2. Medical Oncology Department 1, Fondazione IRCCS - Istituto Nazionale dei Tumori, Milan, Italy

This repository contains the implementation of weighted pathway enrichment strategies for interpretable network-based biomarker discovery in immunotherapy response prediction. We propose two weighted pathway selection methods:

- wORA: Weighted Over-Representation Analysis using Wallenius noncentral hypergeometric distribution
- Pre-ranked GSEA: Gene Set Enrichment Analysis with network propagation scores

Both methods incorporate gene proximity scores (obtained via network propagation on protein-protein interaction networks) into pathway enrichment analysis. Evaluated across 9 immunotherapy-treated cohorts (melanoma, gastric, bladder, and NSCLC), these methods select substantially fewer pathways (median: 8 for wORA, 46 for GSEA) compared to standard ORA (median: 162) while maintaining comparable predictive performance.

## Requirements
- Python 3.13.2
- numpy 2.2.4
- pandas 2.2.3
- scikit-learn 1.6.1
- scipy 1.15.2
- networkx 3.4.2
- gseapy 1.1.9
- shap 0.48.0
- matplotlib 3.10.0
- statsmodels 0.14.4

## Contacts
Arianna Rigamonti - arianna.rigamonti@polimi.it; Francesco Trovò - francesco1.trovo@polimi.it

Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy
