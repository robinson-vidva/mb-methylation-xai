"""Quality control of the methylation data before any modelling.

Runs after 02_preprocess.py (it reads the processed matrices) and writes QC tables and figures:
  qc_probe_funnel.csv          probe counts at each filtering step
  qc_sample_stats.csv          per-sample beta distribution summaries, Cavalli cohort
  qc_external_detection.csv    per-sample fraction of failed probes (detection p > 0.01), Hovestadt cohort
  qc_external_sex_check.csv    mean beta on X chromosome probes against reported sex, Hovestadt cohort
  qc_pca_variance.csv          variance explained by the first principal components
  figures: qc_beta_density.png, qc_sample_medians.png, qc_external_detection.png,
           qc_external_sex_check.png, qc_pca.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from common import FIG, GROUPS, GROUP_COLORS, N_FEATURES, PROC, RAW, SEED, TAB, load_manifest, read_series_matrix

sns.set_theme(style="white", font_scale=0.9)
rng = np.random.default_rng(SEED)


def main():
    lab = pd.read_csv(PROC / "labels.csv", index_col=0)
    man = load_manifest()

    # ---- 1. probe funnel
    cols = pd.read_csv(RAW / "GSE85212_beta.txt.gz", sep="\t", nrows=0).columns.tolist()
    idx = pd.read_csv(RAW / "GSE85212_beta.txt.gz", sep="\t", skiprows=1, usecols=[0],
                      names=["probe"] + cols)["probe"]
    m = man.reindex(idx)
    autosomal = m["CHR"].isin([str(i) for i in range(1, 23)]).values
    snp_free = autosomal & m["Probe_SNPs"].isna().values
    cg_only = snp_free & idx.str.startswith("cg").values
    beta = pd.read_parquet(PROC / "beta_filtered_cache.parquet")
    ext_cols = pd.read_csv(RAW / "GSE54880_beta.txt.gz", sep="\t", nrows=0).columns
    ext_beta_cols = [c for c in ext_cols if c.endswith(".AVG_Beta")]
    ext_pval_cols = [c.replace(".AVG_Beta", ".DetectionPval") for c in ext_beta_cols]
    ext_idx = pd.read_csv(RAW / "GSE54880_beta.txt.gz", sep="\t", usecols=["ID_REF"])["ID_REF"]
    funnel = pd.DataFrame([
        ("probes on the Illumina 450k array (GPL13534 manifest)", len(man)),
        ("probes in the deposited GSE85212 beta matrix", len(idx)),
        ("of these, on autosomes (the submitters had already removed X and Y)", int(autosomal.sum())),
        ("without a known SNP at the probe", int(snp_free.sum())),
        ("cg probes only (ch probes removed)", int(cg_only.sum())),
        ("without any missing value across 763 tumours", int(beta.shape[0])),
        ("also present in the external GSE54880 matrix", int(beta.index.isin(ext_idx).sum())),
        (f"selected by training-set variance for modelling", N_FEATURES),
    ], columns=["step", "probes"])
    funnel.to_csv(TAB / "qc_probe_funnel.csv", index=False)
    print(funnel.to_string(index=False))

    # ---- 2. per-sample beta distribution, Cavalli cohort
    vals = beta.values  # probes x samples, float32
    stats = pd.DataFrame({
        "mean_beta": vals.mean(axis=0),
        "median_beta": np.median(vals, axis=0),
        "sd_beta": vals.std(axis=0),
        "fraction_below_0.2": (vals < 0.2).mean(axis=0),
        "fraction_above_0.8": (vals > 0.8).mean(axis=0),
    }, index=beta.columns)
    stats = stats.join(lab[["group", "subtype", "split"]])
    stats.index.name = "sample"
    stats.round(4).to_csv(TAB / "qc_sample_stats.csv")
    print(stats.groupby("group")[["mean_beta", "fraction_below_0.2", "fraction_above_0.8"]].mean().round(3))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    grid = np.linspace(0, 1, 101)
    for g in GROUPS:
        ids = lab.index[lab["group"] == g]
        pick = rng.choice(ids, size=min(15, len(ids)), replace=False)
        for s in pick:
            h, _ = np.histogram(beta[s].values, bins=grid, density=True)
            ax.plot(grid[:-1] + 0.005, h, color=GROUP_COLORS[g], alpha=0.25, lw=0.8)
        ax.plot([], [], color=GROUP_COLORS[g], label=f"{g} (15 random tumours)")
    ax.set_xlabel("beta value (0 = unmethylated, 1 = methylated)")
    ax.set_ylabel("density")
    ax.set_title("Beta value distribution per tumour, 308,020 CpGs", loc="left")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "qc_beta_density.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    order = GROUPS
    sns.boxplot(data=stats, x="group", y="median_beta", order=order, hue="group", palette=GROUP_COLORS, legend=False,
                ax=axes[0], fliersize=2, width=0.6)
    axes[0].set_title("Median beta per tumour", loc="left")
    sns.boxplot(data=stats, x="group", y="sd_beta", order=order, hue="group", palette=GROUP_COLORS, legend=False,
                ax=axes[1], fliersize=2, width=0.6)
    axes[1].set_title("Standard deviation of beta per tumour", loc="left")
    for ax in axes:
        ax.set_xlabel("")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "qc_sample_medians.png", dpi=160)
    plt.close(fig)

    # ---- 3. external cohort: detection p-values and X chromosome check
    ext = pd.read_csv(RAW / "GSE54880_beta.txt.gz", sep="\t", index_col=0,
                      usecols=["ID_REF"] + ext_beta_cols + ext_pval_cols,
                      dtype={c: np.float32 for c in ext_beta_cols + ext_pval_cols},
                      na_values=[" ", "", "NA"])
    ext_p = ext[ext_pval_cols]
    ext_p.columns = [c.replace(".DetectionPval", "") for c in ext_pval_cols]
    ext_b = ext[ext_beta_cols]
    ext_b.columns = [c.replace(".AVG_Beta", "") for c in ext_beta_cols]
    meta = read_series_matrix(RAW / "GSE54880_series_matrix.txt.gz").set_index("title")
    meta = meta.loc[ext_p.columns]
    det = pd.DataFrame({
        "fraction_failed_probes": (ext_p > 0.01).mean(axis=0).values,
        "fraction_failed_of_modelled_cpgs": (ext_p.reindex(beta.index) > 0.01).mean(axis=0).values,
        "material": meta["sample_type"].map(lambda s: "FFPE" if "FFPE" in s else "frozen").values,
        "group": meta["molecular_subgroup"].values,
        "tissue": meta["tissue"].values,
        "sex": meta["gender"].values,
    }, index=ext_p.columns)
    det.index.name = "sample"
    det.round(4).to_csv(TAB / "qc_external_detection.csv")
    print(det.groupby("material")[["fraction_failed_probes", "fraction_failed_of_modelled_cpgs"]].describe().round(4).T)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.stripplot(data=det, x="material", y="fraction_failed_of_modelled_cpgs", hue="material",
                  palette={"frozen": "#2a78d6", "FFPE": "#eb6834"}, size=3.5, alpha=0.7, ax=ax, legend=False)
    ax.axhline(0.05, color="#888", lw=0.8, ls="--")
    ax.set_ylabel("fraction of the 308,020 CpGs with detection p > 0.01")
    ax.set_xlabel("")
    ax.set_title("Failed probes per sample, external cohort (dashed: 5 percent)", loc="left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "qc_external_detection.png", dpi=160)
    plt.close(fig)

    x_probes = man.index[man["CHR"] == "X"].intersection(ext_b.index)
    sexchk = pd.DataFrame({"mean_beta_chrX": ext_b.loc[x_probes].mean(axis=0).values,
                           "sex": det["sex"].values, "material": det["material"].values},
                          index=ext_b.columns)
    sexchk.index.name = "sample"
    sexchk.round(4).to_csv(TAB / "qc_external_sex_check.csv")
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.stripplot(data=sexchk, x="sex", y="mean_beta_chrX", hue="material",
                  palette={"frozen": "#2a78d6", "FFPE": "#eb6834"}, size=3.5, alpha=0.7, ax=ax, dodge=True)
    ax.set_xlabel("reported sex")
    ax.set_ylabel(f"mean beta over {len(x_probes):,} X chromosome CpGs")
    ax.set_title("Sex check: X inactivation raises female X methylation", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "qc_external_sex_check.png", dpi=160)
    plt.close(fig)
    print(sexchk.groupby("sex")["mean_beta_chrX"].describe().round(3))

    # ---- 4. PCA on the modelled CpGs, fitted on the training split
    X = pd.read_parquet(PROC / "beta_top.parquet")
    lab = lab.loc[X.index]
    tr = (lab["split"] == "train").values
    pca = PCA(n_components=10, random_state=SEED).fit(X.values[tr])
    Z = pca.transform(X.values)
    ext_top = pd.read_parquet(PROC / "external_beta_top.parquet")[X.columns]
    Zext = pca.transform(ext_top.values)
    ext_lab = pd.read_csv(PROC / "external_labels.csv", index_col=0).loc[ext_top.index]
    var = pd.DataFrame({"component": np.arange(1, 11), "variance_explained": pca.explained_variance_ratio_})
    var.round(4).to_csv(TAB / "qc_pca_variance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for g in GROUPS:
        for split, mk in [("train", "o"), ("test", "^")]:
            sel = ((lab["group"] == g) & (lab["split"] == split)).values
            ax.scatter(Z[sel, 0], Z[sel, 1], s=14, marker=mk, color=GROUP_COLORS[g],
                       edgecolor="white", linewidth=0.3, label=f"{g} {split}")
    ax.set_title("Cavalli 2017, 763 tumours (circle train, triangle test)", loc="left", fontsize=10)
    ax = axes[1]
    for g in GROUPS:
        sel = (lab["group"] == g).values
        ax.scatter(Z[sel, 0], Z[sel, 1], s=8, color=GROUP_COLORS[g], alpha=0.25, linewidths=0)
    for mat, mk in [("frozen", "o"), ("FFPE", "s")]:
        for g in GROUPS:
            sel = ((ext_lab["material"] == mat) & (ext_lab["group"] == g)).values
            ax.scatter(Zext[sel, 0], Zext[sel, 1], s=16, marker=mk, color=GROUP_COLORS[g],
                       edgecolor="#222", linewidth=0.4, label=f"{g} {mat}")
    ax.set_title("Hovestadt 2013 projected (circle frozen, square FFPE; grey: Cavalli)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "qc_pca.png", dpi=160)
    plt.close(fig)
    print(var.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
