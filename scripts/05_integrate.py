"""Link methylation to expression, pathways and known driver genes.

Part A  methylation versus expression for the model's top genes (same tumours, Spearman)
Part B  per-group differential promoter methylation and differential expression, concordant genes
Part C  pathway enrichment of concordant genes (g:Profiler web service, GO:BP, KEGG, Reactome)
Part D  overlap with a curated list of known medulloblastoma driver genes
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from common import FIG, GROUPS, GROUP_COLORS, PROC, TAB

sns.set_theme(style="white", font_scale=0.9)

# Curated from Northcott et al. 2017 Nature (doi 10.1038/nature22973) and
# Cavalli et al. 2017 Cancer Cell (doi 10.1016/j.ccell.2017.05.005). Small, well-known subset.
DRIVERS = {
    "WNT": ["CTNNB1", "DDX3X", "SMARCA4", "TP53", "KMT2D"],
    "SHH": ["PTCH1", "SUFU", "SMO", "GLI2", "MYCN", "TP53", "TERT", "KMT2D", "DDX3X"],
    "Group3": ["MYC", "GFI1", "GFI1B", "OTX2", "SMARCA4", "KBTBD4"],
    "Group4": ["KDM6A", "PRDM6", "SNCAIP", "ZMYM3", "KBTBD4", "MYCN", "OTX2", "CDK6"],
}
FDR = 0.05
DELTA_BETA = 0.10
LOG2FC = 1.0


def differential(mat, mask):
    """Mann-Whitney U per column, group (mask) versus rest. Returns delta of means and FDR."""
    a, b = mat.values[mask], mat.values[~mask]
    stat, p = mannwhitneyu(a, b, axis=0)
    q = multipletests(p, method="fdr_bh")[1]
    return pd.DataFrame({"delta": a.mean(0) - b.mean(0), "p": p, "fdr": q}, index=mat.columns)


def gprofiler(query_by_group, background):
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {"organism": "hsapiens", "query": query_by_group, "background": background,
               "domain_scope": "custom", "sources": ["GO:BP", "KEGG", "REAC"],
               "user_threshold": 0.05, "significance_threshold_method": "g_SCS",
               "no_evidences": True}
    for attempt in range(3):
        r = requests.post(url, json=payload, timeout=120)
        if r.ok:
            res = pd.DataFrame(r.json()["result"])
            return res
        time.sleep(5)
    r.raise_for_status()


def main():
    lab = pd.read_csv(PROC / "labels.csv", index_col=0)
    prom = pd.read_parquet(PROC / "promoter_meth.parquet").loc[lab.index]
    # the 2011 Illumina manifest uses older symbols for a few driver genes
    prom = prom.rename(columns={"MLL2": "KMT2D", "MLL3": "KMT2C"})
    expr = pd.read_parquet(PROC / "expr.parquet").loc[lab.index]
    genes = prom.columns.intersection(expr.columns)
    prom, expr = prom[genes], expr[genes]
    print("genes with promoter methylation and expression:", len(genes))

    # ---- Part A: correlation for all genes, then the model's top genes
    rho = np.array([spearmanr(prom[g], expr[g]).statistic for g in genes])
    corr = pd.DataFrame({"gene": genes, "spearman_rho": rho}).set_index("gene")
    corr.round(4).to_csv(TAB / "promoter_methylation_vs_expression_all_genes.csv")
    top = pd.read_csv(TAB / "top_cpgs_per_group.csv")
    top_genes = top[top["gene"].isin(genes)].copy()
    top_genes["spearman_rho"] = corr.loc[top_genes["gene"], "spearman_rho"].values
    top_genes["promoter_cpg"] = top_genes["region"].isin(["TSS200", "TSS1500", "5'UTR", "1stExon"])
    top_genes.round(4).to_csv(TAB / "top_cpgs_with_expression_correlation.csv", index=False)
    summary = {
        "n_genes_tested": int(len(genes)),
        "all_genes_median_rho": float(np.nanmedian(rho)),
        "all_genes_fraction_rho_below_-0.3": float(np.nanmean(rho < -0.3)),
        "top_genes_n": int(top_genes["gene"].nunique()),
        "top_genes_median_rho": float(top_genes.drop_duplicates("gene")["spearman_rho"].median()),
        "top_promoter_genes_n": int(top_genes[top_genes["promoter_cpg"]]["gene"].nunique()),
        "top_promoter_genes_median_rho": float(
            top_genes[top_genes["promoter_cpg"]].drop_duplicates("gene")["spearman_rho"].median()),
        "top_promoter_genes_fraction_rho_below_-0.3": float(
            (top_genes[top_genes["promoter_cpg"]].drop_duplicates("gene")["spearman_rho"] < -0.3).mean()),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rho[~np.isnan(rho)], bins=60, color="#c3c2b7", density=True, label="all genes")
    tg = top_genes[top_genes["promoter_cpg"]].drop_duplicates("gene")["spearman_rho"]
    ax.hist(tg, bins=30, color="#2a78d6", alpha=0.75, density=True,
            label="genes under the model's top promoter CpGs")
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Spearman rho, promoter methylation vs expression (763 tumours)")
    ax.set_ylabel("density")
    ax.set_title("Promoter methylation tracks expression for the model's genes", loc="left")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "methylation_vs_expression_rho.png", dpi=160)
    plt.close(fig)

    # ---- Part B: differential promoter methylation and expression per group
    conc_rows, diff_tables = [], {}
    for g in GROUPS:
        mask = (lab["group"] == g).values
        dm = differential(prom, mask).add_prefix("meth_")
        de = differential(expr, mask).add_prefix("expr_")
        d = dm.join(de)
        d["group"] = g
        d["concordant"] = (
            (d["meth_fdr"] < FDR) & (d["expr_fdr"] < FDR) & (d["meth_delta"].abs() > DELTA_BETA)
            & (d["expr_delta"].abs() > LOG2FC) & (np.sign(d["meth_delta"]) != np.sign(d["expr_delta"])))
        d["pattern"] = np.where(d["meth_delta"] > 0, "promoter hypermethylated, expression down",
                                "promoter hypomethylated, expression up")
        diff_tables[g] = d
        conc_rows.append(d[d["concordant"]].reset_index().rename(columns={"index": "gene"}))
    diff_all = pd.concat(diff_tables.values()).reset_index().rename(columns={"index": "gene"})
    diff_all.round(5).to_csv(TAB / "differential_promoter_methylation_expression.csv", index=False)
    conc = pd.concat(conc_rows)
    conc.round(5).to_csv(TAB / "concordant_genes_per_group.csv", index=False)
    summary["concordant_genes_per_group"] = conc.groupby("group").size().to_dict()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, g in zip(axes, GROUPS):
        d = diff_tables[g]
        ax.scatter(d["meth_delta"], d["expr_delta"], s=4, color="#c3c2b7", alpha=0.5, linewidths=0)
        c = d[d["concordant"]]
        ax.scatter(c["meth_delta"], c["expr_delta"], s=10, color=GROUP_COLORS[g], linewidths=0)
        for gene in set(DRIVERS[g]) & set(d.index):
            ax.annotate(gene, (d.loc[gene, "meth_delta"], d.loc[gene, "expr_delta"]), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
            ax.scatter([d.loc[gene, "meth_delta"]], [d.loc[gene, "expr_delta"]], s=24,
                       facecolor="none", edgecolor="#222", linewidths=0.8)
        ax.axhline(0, color="#444", lw=0.6)
        ax.axvline(0, color="#444", lw=0.6)
        ax.set_title(f"{g} vs rest, {int(d['concordant'].sum())} concordant genes", loc="left", fontsize=10)
        ax.set_xlabel("delta promoter beta")
    axes[0].set_ylabel("delta log2 expression")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "differential_meth_vs_expr.png", dpi=160)
    plt.close(fig)

    # ---- Part C: pathway enrichment of concordant genes
    query = {g: conc.loc[conc["group"] == g, "gene"].tolist() for g in GROUPS}
    query = {g: v for g, v in query.items() if len(v) >= 5}
    enr = gprofiler(query, list(genes))
    if len(enr):
        enr = enr[["query", "source", "native", "name", "p_value", "term_size", "query_size",
                   "intersection_size", "precision", "recall"]].sort_values(["query", "p_value"])
        enr.to_csv(TAB / "pathway_enrichment_concordant_genes.csv", index=False)
        topk = enr[enr["term_size"] <= 500].groupby("query").head(8).copy()
        topk["label"] = topk["name"].str.slice(0, 55) + " (" + topk["source"] + ")"
        topk["neglog10p"] = -np.log10(topk["p_value"])
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(topk) + 1.5))
        y = np.arange(len(topk))[::-1]
        ax.scatter(topk["neglog10p"], y, s=topk["intersection_size"] * 6,
                   color=[GROUP_COLORS[q] for q in topk["query"]], linewidths=0)
        ax.set_yticks(y)
        ax.set_yticklabels(topk["label"], fontsize=8)
        ax.set_xlabel("-log10 adjusted p (g:SCS)")
        ax.set_title("Pathways enriched in concordant genes (colour = group, size = genes in term)",
                     loc="left", fontsize=10)
        for g in GROUPS:
            if g in topk["query"].values:
                ax.scatter([], [], color=GROUP_COLORS[g], label=g)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        sns.despine()
        fig.tight_layout()
        fig.savefig(FIG / "pathway_enrichment.png", dpi=160)
        plt.close(fig)
    summary["enriched_terms_per_group"] = enr.groupby("query").size().to_dict() if len(enr) else {}

    # ---- Part D: known drivers
    top_gene_sets = {g: set(top.loc[top["group"] == g, "gene"].dropna()) for g in GROUPS}
    rows = []
    for g, dl in DRIVERS.items():
        d = diff_tables[g]
        for gene in dl:
            rows.append({
                "group": g, "driver_gene": gene,
                "in_model_top50_cpgs": gene in top_gene_sets[g],
                "has_array_data": gene in d.index,
                "delta_promoter_beta": d.loc[gene, "meth_delta"] if gene in d.index else np.nan,
                "meth_fdr": d.loc[gene, "meth_fdr"] if gene in d.index else np.nan,
                "delta_log2_expression": d.loc[gene, "expr_delta"] if gene in d.index else np.nan,
                "expr_fdr": d.loc[gene, "expr_fdr"] if gene in d.index else np.nan,
                "spearman_rho_meth_expr": corr.loc[gene, "spearman_rho"] if gene in corr.index else np.nan,
                "concordant": bool(d.loc[gene, "concordant"]) if gene in d.index else False,
            })
    drivers = pd.DataFrame(rows)
    drivers.round(4).to_csv(TAB / "driver_gene_overlap.csv", index=False)
    summary["drivers_in_top50"] = int(drivers["in_model_top50_cpgs"].sum())
    summary["drivers_differentially_expressed_fdr05"] = int((drivers["expr_fdr"] < FDR).sum())
    json.dump(summary, open(TAB / "integration_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(drivers[["group", "driver_gene", "in_model_top50_cpgs", "delta_promoter_beta",
                   "delta_log2_expression", "expr_fdr", "concordant"]])


if __name__ == "__main__":
    main()
