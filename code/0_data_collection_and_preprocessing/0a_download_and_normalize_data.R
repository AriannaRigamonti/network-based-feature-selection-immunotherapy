################################################################################
# Script: 0a_download_and_normalize_data.R
# Author: Arianna Rigamonti
# Description: Process and normalize mRNA expression and metadata 
#              from multiple public immunotherapy datasets.
#              For each dataset, metadata and normalized expression matrices 
#              are exported as tab-delimited text files for downstream analyses.
################################################################################

setwd("path_to_your_directory") # Set working directory

### Dataset: Gide (Melanoma, anti–PD1/CTLA4)

# Load raw data object
load("dat.Gide.RData")
output_dir <- "txt_output/Gide"

# Extract and save metadata
meta_df <- data.frame(
  Patient = dat$samples,
  Age = dat$age,
  Sex = dat$sex,
  Treatment = dat$treatment,
  Response = dat$response,
  Flag = dat$flag,
  OS = dat$survival$time,
  Status = dat$survival$status
)

# Convert clinical response categories into binary labels
meta_df$Response <- ifelse(meta_df$Response %in% c("CR", "PR"), 1,
                       ifelse(meta_df$Response %in% c("SD", "PD"), 0, NA))

write.table(meta_df, file = file.path(output_dir, "Gide_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Save expression matrix with gene and sample annotations
save_matrix <- function(mat, name) {
  df <- as.data.frame(mat)
  rownames(df) <- dat$genes
  colnames(df) <- dat$samples
  df <- cbind(gene_id = rownames(df), df)  
  write.table(df, file = file.path(output_dir, paste0("Gide_", name, ".txt")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

save_matrix(dat$mRNA, "mRNA")


### Dataset: Liu (Melanoma, anti–PD1)

# Load raw data object
load("dat.Liu.RData")
output_dir <- "txt_output/Liu"

# Extract and save metadata
meta_df <- data.frame(
  Patient = dat$samples,
  Sex = dat$sex,
  Treatment = dat$treatment,
  Response = dat$response,
  OS = dat$surv.dt$time,
  Status = dat$surv.dt$status
)

# Convert clinical response categories into binary labels
meta_df$Response <- ifelse(meta_df$Response %in% c("CR", "PR", "MR"), 1,
                           ifelse(meta_df$Response %in% c("SD", "PD"), 0, NA))

write.table(meta_df, file = file.path(output_dir, "Liu_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Save expression matrix
save_matrix <- function(mat, name) {
  df <- as.data.frame(mat)
  rownames(df) <- make.unique(dat$genes)
  colnames(df) <- dat$samples
  df <- cbind(gene_id = rownames(df), df)  
  write.table(df, file = file.path(output_dir, paste0("Liu_", name, ".txt")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

save_matrix(dat$mRNA, "mRNA")


### Dataset: Kim (Metastatic Gastric Cancer, anti–PD1)

# Load raw data object
load("dat.Kim.RData")
output_dir <- "txt_output/Kim"

# Extract and save metadata
meta_df <- data.frame(
  Patient = dat$samples,
  Response = dat$response
)

write.table(meta_df, file = file.path(output_dir, "Kim_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Save expression matrix
save_matrix <- function(mat, name) {
  df <- as.data.frame(mat)
  gene_id <- as.character(dat$genes)
  gene_id[is.na(gene_id) | gene_id == ""] <- paste0("Unknown_", seq_len(sum(is.na(gene_id) | gene_id == "")))
  rownames(df) <- make.unique(gene_id)
  colnames(df) <- dat$samples
  df <- cbind(gene_id = rownames(df), df)
  write.table(df,
              file = file.path(output_dir, paste0("Kim_", name, ".txt")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

save_matrix(dat$mRNA.norm3, "mRNA.norm3")


### Dataset: Huang (Melanoma, anti–PD1)

# Load raw data object
load("dat.Huang.RData")
output_dir <- "txt_output/Huang"

# Extract and save metadata
meta_df <- data.frame(
  Patient = dat$samples,
  Response = dat$response
)

write.table(meta_df, file = file.path(output_dir, "Huang_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Save cleaned expression matrix
expr_data <- dat$mRNA
gene_id <- as.character(dat$genes)
rownames(expr_data) <- make.unique(gene_id)
colnames(expr_data) <- dat$samples
expr_data <- expr_data[complete.cases(expr_data), ]
min_val <- min(expr_data[is.finite(expr_data)], na.rm = TRUE)
expr_data[expr_data == -Inf] <- min_val

expr_out <- cbind(gene_id = rownames(expr_data), expr_data)
write.table(
  expr_out,
  file = file.path(output_dir, "Huang_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)


### Dataset: Auslander (Melanoma, anti–PD1 and/or anti–CTLA4)

# Metadata manually extracted from NIHMS1032642 supplementary table in:
# Auslander et al., Nat Med. 2018;24(10):1545–1549. doi:10.1038/s41591-018-0157-9

# Process and normalize raw count data
library(edgeR)
output_dir <- "txt_output/Auslander"
data <- read.csv2("Auslander_raw_counts.csv", sep = ";")
gene_id <- data[[1]]
counts  <- data[ , -1, drop = FALSE]

counts_mat <- as.matrix(sapply(counts, function(x) as.numeric(x)))
rownames(counts_mat) <- make.unique(as.character(gene_id))

# TMM normalization using edgeR
dge_tmp <- DGEList(counts = counts_mat)
dge <- DGEList(counts = dge_tmp)
dge <- calcNormFactors(dge, method = "TMM")

# Compute CPM normalized expression matrix
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)

# Export normalized data
write.table(
  cbind(gene_id = rownames(cpm_tmm), cpm_tmm),
  file = file.path(output_dir, "Auslander_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)


### Dataset: Riaz (Melanoma, anti–PD1)

# Metadata available from the original paper GitHub repository:
# https://github.com/riazn/bms038_analysis/blob/master/data/bms038_clinical_data.csv

# Process and normalize expression data
library(edgeR)
library(org.Hs.eg.db)
library(AnnotationDbi)
output_dir <- "txt_output/Riaz"

data <- read.csv2("Riaz_raw.tsv", sep = "\t")
gene_id <- as.character(data[[1]])

# Map Entrez IDs to gene symbols
map <- AnnotationDbi::select(org.Hs.eg.db,
                             keys = gene_id,
                             keytype = "ENTREZID",
                             columns = c("SYMBOL","GENENAME"))
new_gene_id <- map$SYMBOL
counts  <- data[ , -1, drop = FALSE]

# Keep valid gene symbols and normalize counts
counts_mat <- as.matrix(sapply(counts, function(x) as.numeric(x)))
keep <- !is.na(new_gene_id) & new_gene_id != ""
counts_mat <- counts_mat[keep, , drop = FALSE]
rownames(counts_mat) <- make.unique(as.character(new_gene_id[keep]))

# TMM normalization
dge_tmp <- DGEList(counts = counts_mat)
dge <- DGEList(counts = dge_tmp)
dge <- calcNormFactors(dge, method = "TMM")

cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)

# Export normalized data
write.table(
  cbind(gene_id = rownames(cpm_tmm), cpm_tmm),
  file = file.path(output_dir, "Riaz_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)


### Dataset: Prat (Melanoma, anti–PD1)

# Metadata retrieved from the original publication (Supplementary Table 1)
library(edgeR)
output_dir <- "txt_output/Prat"
data <- readxl::read_excel("Prat_raw.xls")
gene_id <- data[[1]]
counts  <- data[ , -1, drop = FALSE]

# Select melanoma samples only
keep_ids <- c("A41","A42","A43","A44","A45","A46","A47","A48","A49","A50",
              "A51","A52","A53","A54","A55","A56","A57","A58","A59","A60",
              "A61","A62","A63","A64","A65")
present_ids <- intersect(keep_ids, colnames(counts))
counts <- counts[ , present_ids, drop = FALSE]

# TMM normalization
counts_mat <- as.matrix(sapply(counts, function(x) as.numeric(x)))
rownames(counts_mat) <- make.unique(as.character(gene_id))
dge_tmp <- DGEList(counts = counts_mat)
dge <- DGEList(counts = dge_tmp)
dge <- calcNormFactors(dge, method = "TMM")
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)

# Export normalized data
write.table(
  cbind(gene_id = rownames(cpm_tmm), cpm_tmm),
  file = file.path(output_dir, "Prat_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

### Dataset: IMvigor210 (Urothelial Carcinoma, anti–PDL1)

# Data obtained from the official repository:
# http://research-pub.gene.com/IMvigor210CoreBiologies/

devtools::install_github('PiotrTymoszuk/DESeq') # install dependencies if required
install.packages("IMvigor210CoreBiologies_1.0.0.tar.gz", repos = NULL)

library(IMvigor210CoreBiologies) 
library(tidyverse) 
library(org.Hs.eg.db)
library(AnnotationDbi)
library(edgeR)
output_dir <- "txt_output/IMvigor210"
data(cds)

# Extract and save metadata
pheno <- cds@phenoData@data %>% as.data.frame()
pheno2 <- pheno %>%
  mutate(
    Response = case_when(
      `Best Confirmed Overall Response` %in% c("CR", "PR") ~ 1L,
      `Best Confirmed Overall Response` %in% c("SD", "PD") ~ 0L,
      TRUE ~ NA_integer_
    )
  )
pheno2 <- pheno2[!is.na(pheno2$Response), ]
pheno2 <- pheno2 %>%
  rownames_to_column(var = "sample_id")

write.table(pheno2, file = file.path(output_dir, "IMvigor_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Process and normalize expression counts
assay_env <- cds@assayData
cts <- get("counts", envir = assay_env)

gene_id <- row.names(cts)
map <- AnnotationDbi::select(org.Hs.eg.db,
                             keys = gene_id,
                             keytype = "ENTREZID",
                             columns = c("SYMBOL","GENENAME"))
new_gene_id <- map$SYMBOL
row.names(cts) <- new_gene_id

counts_mat <- as.matrix(cts)
keep <- !is.na(new_gene_id) & new_gene_id != ""
counts_mat <- counts_mat[keep, , drop = FALSE]
rownames(counts_mat) <- make.unique(as.character(new_gene_id[keep]))

# Match metadata and expression samples
expr_ids <- colnames(counts_mat)
keep_ids <- pheno2$sample_id
samples_in_both <- intersect(expr_ids, keep_ids)
counts_mat  <- counts_mat[, samples_in_both, drop = FALSE]

# TMM normalization
dge_tmp <- DGEList(counts = counts_mat)
dge <- DGEList(counts = dge_tmp)
dge <- calcNormFactors(dge, method = "TMM")
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)

# Export normalized data
write.table(
  cbind(gene_id = rownames(cpm_tmm), cpm_tmm),
  file = file.path(output_dir, "IMvigor_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)


### Dataset: Ravi (NSCLC, anti–PD1/PDL1/CTLA4)

# Data obtained from:
# Ravi et al., Nat Genet 55, 807–819 (2023). doi:10.1038/s41588-023-01355-5

library(readxl)
library(stringr)
library(dplyr)
library(edgeR)
output_dir <- "txt_output/Ravi"

# Process and clean metadata
metadata <- readxl::read_excel("Ravi_raw_clinical_annotations.xlsx")
true_header <- as.character(unlist(metadata[2, ]))
true_header[is.na(true_header)] <- paste0("Unnamed_", which(is.na(true_header)))
true_header <- str_replace_all(true_header, "\\s+$", "")
true_header <- str_replace_all(true_header, "\\s+", "_")

dat <- metadata[-c(1, 2), , drop = FALSE]
colnames(dat) <- true_header
dat <- dat |>
  tibble::as_tibble() |>
  mutate(across(where(is.character), ~str_trim(.x)))

# Reformat column names and select relevant entries
dat$Center <- dat$Institution 
dat$Drug <- dat$Agent_PD1
dat$Drug_category <- dat$Agent_PD1_Category
dat$Sequencing <- dat$Sequencing_Platform
dat$Histology <- dat$Histology_Harmonized
dat$Therapy_lines <- dat$Line_of_Therapy
dat <- dat[, 11:ncol(dat)]
dat <- dat[!is.na(dat[[1]]), ]

# Keep patients treated with PD(L)1 or PD(L)1 + CTLA4
dat <- dat %>%
  filter(Agent_PD1_Category %in% c("PD(L)1", "PD(L)1 + CTLA4"))

# Convert clinical response categories to binary labels
dat$Response <- ifelse(dat$Harmonized_Confirmed_BOR %in% c("CR", "PR"), 1,
                       ifelse(dat$Harmonized_Confirmed_BOR %in% c("SD", "PD"), 0, NA))
dat <- dat[!is.na(dat$Response), ]

# Export metadata
write.table(dat, file = file.path(output_dir, "Ravi_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Process and normalize expression data
data <- readxl::read_excel("Ravi_raw.xlsx")
gene_id <- data[[1]]
counts  <- data[ , -1, drop = FALSE]

rna_id_col <- "Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"
keep_rna_ids <- dat %>%
  pull(all_of(rna_id_col)) %>%
  unique() %>%
  na.omit() %>%
  as.character()

expr_sample_names <- colnames(counts)
samples_in_both <- intersect(keep_rna_ids, expr_sample_names)
counts_subset <- counts[, samples_in_both, drop = FALSE]

# TMM normalization
counts_mat <- as.matrix(sapply(counts_subset, function(x) as.numeric(x)))
rownames(counts_mat) <- make.unique(as.character(gene_id))
dge_tmp <- DGEList(counts = counts_mat)
dge <- DGEList(counts = dge_tmp)
dge <- calcNormFactors(dge, method = "TMM")
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)

# Export normalized data
write.table(
  cbind(gene_id = rownames(cpm_tmm), cpm_tmm),
  file = file.path(output_dir, "Ravi_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)


### Dataset: PEOPLE (NSCLC, anti–PD1)

# Accession restricted dataset, processed from local files
library(edgeR)
output_dir <- "txt_output/PEOPLE"

# Load metadata and raw count matrix
dat <- readxl::read_excel("PEOPLE_raw_clinical_annotations.xlsx")
data <- read_csv2("PEOPLE_raw.csv")
gene_id <- data[[1]]
counts  <- data[ , -1, drop = FALSE]

# Match metadata and expression sample IDs
expr_sample <- colnames(counts)
meta_sample <- dat$ID
keep_ids <- intersect(expr_sample, meta_sample)

# Filter metadata by matched IDs
dat_sub <- dat %>%
  filter(ID %in% keep_ids) %>%
  arrange(match(ID, expr_sample))

# Convert response to binary labels
dat_sub$Response <- ifelse(dat_sub$best_response %in% c("CR", "RP"), 1,
                       ifelse(dat_sub$best_response %in% c("SD", "PD"), 0, NA))
dat_sub <- dat_sub[!is.na(dat_sub$Response), ]

# Export metadata
write.table(dat_sub, file = file.path(output_dir, "PEOPLE_metadata.txt"),
            sep = "\t", quote = FALSE, row.names = FALSE)

meta_samples <- dat_sub$ID

# Process and normalize expression data
counts_sub <- counts[, meta_samples, drop = FALSE]
counts_mat <- as.matrix(sapply(counts_sub, function(x) as.numeric(x)))
rownames(counts_mat) <- make.unique(as.character(gene_id))

# TMM normalization
dge_tmp <- DGEList(counts = counts_mat)
dge <- DGEList(counts = dge_tmp)
dge <- calcNormFactors(dge, method = "TMM")
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)

# Export normalized data
write.table(
  cbind(gene_id = rownames(cpm_tmm), cpm_tmm),
  file = file.path(output_dir, "PEOPLE_mRNA.txt"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
