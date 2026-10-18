"""Build the analysis matrices from the raw GEO files.

Outputs in data/processed:
  labels.csv            sample, group, subtype, split (train/test)
  beta_top.parquet      samples x N_FEATURES CpGs (selected on the training set only)
  feature_rank.csv      all retained CpGs ranked by training-set variance
  promoter_meth.parquet samples x genes, mean beta over promoter probes
  body_meth.parquet     samples x genes, mean beta over gene body probes
  expr.parquet          samples x genes, log2 expression (GSE85217)
  external_beta_top.parquet  GSE54880 samples x the same N_FEATURES CpGs
  external_labels.csv   GSE54880 sample, group, material (FFPE or frozen)
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from common import (BODY_REGIONS, N_FEATURES, PROC, PROMOTER_REGIONS, RAW, SEED,
                    load_manifest, probe_gene_pairs, read_series_matrix)


def gene_level_mean(beta, pairs):
    """Mean beta per gene over the probes listed in pairs (probe, gene)."""
    pairs = pairs[pairs["probe"].isin(beta.index)]
    probe_idx = pd.Index(beta.index).get_indexer(pairs["probe"])
    genes = pd.Index(sorted(pairs["gene"].unique()))
    gene_idx = genes.get_indexer(pairs["gene"])
    ind = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (gene_idx, probe_idx)),
                        shape=(len(genes), beta.shape[0]))
    counts = np.asarray(ind.sum(axis=1)).ravel()
    sums = ind @ beta.values
    return pd.DataFrame((sums / counts[:, None]).T, index=beta.columns, columns=genes)


def main():
    # ---- labels, Cavalli cohort
    meta = read_series_matrix(RAW / "GSE85212_series_matrix.txt.gz")
    meta["sample"] = meta["title"].str.replace("_methylation", "", regex=False)
    labels = meta[["sample", "subgroup", "subtype_formatted"]].rename(
        columns={"subtype_formatted": "subtype", "subgroup": "group"}).set_index("sample")
    train_ids, test_ids = train_test_split(labels.index.to_numpy(), test_size=0.2,
                                           stratify=labels["subtype"].to_numpy(),
                                           random_state=SEED)
    train_ids, test_ids = list(train_ids), list(test_ids)
    labels["split"] = "train"
    labels.loc[test_ids, "split"] = "test"

    # ---- beta matrix, 485k probes x 763 samples
    man = load_manifest()
    cache = PROC / "beta_filtered_cache.parquet"
    if cache.exists():
        beta = pd.read_parquet(cache)
    else:
        print("reading beta matrix")
        cols = pd.read_csv(RAW / "GSE85212_beta.txt.gz", sep="\t", nrows=0).columns.tolist()
        beta = pd.read_csv(RAW / "GSE85212_beta.txt.gz", sep="\t", skiprows=1, index_col=0,
                           names=["probe"] + cols, dtype={c: np.float32 for c in cols})
        beta = beta[labels.index]
        print("beta", beta.shape, "missing fraction", float(np.isnan(beta.values).mean()))
        # autosomal, SNP-free cg probes without missing values
        keep = man.index[(man["CHR"].isin([str(i) for i in range(1, 23)]))
                         & man["Probe_SNPs"].isna()
                         & man.index.str.startswith("cg")]
        beta = beta.loc[beta.index.intersection(keep)].dropna(axis=0)
        beta.to_parquet(cache)
    print("after filtering autosomal, SNP-free, complete probes:", beta.shape)

    # ---- external cohort probes must exist too
    ext_cols = pd.read_csv(RAW / "GSE54880_beta.txt.gz", sep="\t", nrows=0).columns
    ext_beta_cols = [c for c in ext_cols if c.endswith(".AVG_Beta")]
    ext_pval_cols = [c.replace(".AVG_Beta", ".DetectionPval") for c in ext_beta_cols]
    ext = pd.read_csv(RAW / "GSE54880_beta.txt.gz", sep="\t", index_col=0,
                      usecols=["ID_REF"] + ext_beta_cols + ext_pval_cols,
                      dtype={c: np.float32 for c in ext_beta_cols + ext_pval_cols},
                      na_values=[" ", "", "NA"])
    ext_b = ext[ext_beta_cols].copy()
    ext_b.columns = [c.replace(".AVG_Beta", "") for c in ext_beta_cols]
    ext_p = ext[ext_pval_cols].copy()
    ext_p.columns = ext_b.columns
    ext_b = ext_b.mask(ext_p > 0.01)
    common = beta.index.intersection(ext_b.index)
    beta = beta.loc[common]
    print("probes shared with external cohort:", len(common))

    # ---- gene level summaries on the full retained probe set
    print("gene level means")
    gene_level_mean(beta, probe_gene_pairs(man, PROMOTER_REGIONS)).to_parquet(PROC / "promoter_meth.parquet")
    gene_level_mean(beta, probe_gene_pairs(man, BODY_REGIONS)).to_parquet(PROC / "body_meth.parquet")

    # ---- feature selection on the training set only
    var = beta[train_ids].var(axis=1)
    rank = var.sort_values(ascending=False)
    rank.rename("train_variance").to_csv(PROC / "feature_rank.csv")
    top = rank.index[:N_FEATURES]
    beta.loc[top].T.to_parquet(PROC / "beta_top.parquet")

    # ---- external cohort on the same features, median imputation from training data
    ext_top = ext_b.loc[top].T
    ext_top = ext_top.fillna(beta.loc[top, train_ids].median(axis=1))
    ext_top.to_parquet(PROC / "external_beta_top.parquet")
    ext_meta = read_series_matrix(RAW / "GSE54880_series_matrix.txt.gz").set_index("title")
    ext_labels = pd.DataFrame({
        "group": ext_meta.loc[ext_top.index, "molecular_subgroup"].replace(
            {"Group 3": "Group3", "Group 4": "Group4"}),
        "material": ext_meta.loc[ext_top.index, "sample_type"].map(
            lambda s: "FFPE" if "FFPE" in s else "frozen"),
        "tissue": ext_meta.loc[ext_top.index, "tissue"],
    })
    ext_labels.index.name = "sample"
    ext_labels.to_csv(PROC / "external_labels.csv")

    # ---- expression, same 763 samples
    print("reading expression")
    expr = pd.read_csv(RAW / "GSE85217_expr.txt.gz", sep="\t", index_col=0)
    sym = expr["HGNC_symbol_from_ensemblv77"]
    expr = expr[labels.index].astype(np.float32)
    expr = expr[sym.notna().values]
    expr.index = sym[sym.notna()].values
    expr = expr.groupby(level=0).mean()  # a few symbols map to several probe sets
    expr.T.to_parquet(PROC / "expr.parquet")

    labels.to_csv(PROC / "labels.csv")
    print("done. train", (labels["split"] == "train").sum(), "test", (labels["split"] == "test").sum())
    print(labels.groupby(["group", "subtype"]).size())


if __name__ == "__main__":
    main()
