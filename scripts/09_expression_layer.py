"""The expression layer on its own: what differential expression adds beyond the concordant genes.

For every gene and subgroup (from 05_integrate.py) the change is placed in one of five categories:
  concordant            promoter methylation and expression both changed, opposite directions
  expression only       expression changed (FDR < 0.05, |log2 fold change| > 1), promoter not
  methylation only      promoter changed (FDR < 0.05, |delta beta| > 0.10), expression not
  both, same direction  both changed but in the same direction (not the canonical promoter logic)
  neither
Then, for the expression-only genes: g:Profiler enrichment per subgroup, oncogene and tumour
suppressor status (OncoKB), overlap with the curated drivers, and overlap with the 22-gene
NanoString subgrouping signature of Northcott et al. 2012.

Outputs in results/tables:
  expression_layer_counts.csv, expression_only_genes.csv, expression_layer_enrichment.csv,
  nanostring_signature_check.csv, expression_layer_summary.json
Figures: expression_layer_scatter.png, expression_only_top_genes.png
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

from common import FIG, GROUPS, GROUP_COLORS, TAB

sns.set_theme(style="white", font_scale=0.9)
FDR, DELTA_BETA, LOG2FC = 0.05, 0.10, 1.0
CATS = ["concordant", "expression only", "methylation only", "both, same direction", "neither"]
DRIVERS = {
    "WNT": ["CTNNB1", "DDX3X", "SMARCA4", "TP53", "KMT2D"],
    "SHH": ["PTCH1", "SUFU", "SMO", "GLI2", "MYCN", "TP53", "TERT", "KMT2D", "DDX3X"],
    "Group3": ["MYC", "GFI1", "GFI1B", "OTX2", "SMARCA4", "KBTBD4"],
    "Group4": ["KDM6A", "PRDM6", "SNCAIP", "ZMYM3", "KBTBD4", "MYCN", "OTX2", "CDK6"],
}
# Northcott et al. 2012, Acta Neuropathol, 22-gene NanoString signature (EGFL11 is now EYS).
NANOSTRING = {
    "WNT": ["WIF1", "TNC", "GAD1", "DKK2", "EMX2"],
    "SHH": ["PDLIM3", "EYA1", "HHIP", "ATOH1", "SFRP1"],
    "Group3": ["IMPG2", "GABRA5", "EYS", "NRL", "MAB21L2", "NPR3"],
    "Group4": ["KCNA1", "EOMES", "KHDRBS2", "RBM24", "UNC5D", "OAS1"],
}


def gprofiler(query_by_name, background):
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {"organism": "hsapiens", "query": query_by_name, "background": background,
               "domain_scope": "custom", "sources": ["GO:BP", "KEGG", "REAC"],
               "user_threshold": 0.05, "significance_threshold_method": "g_SCS", "no_evidences": True}
    for attempt in range(3):
        r = requests.post(url, json=payload, timeout=180)
        if r.ok:
            return pd.DataFrame(r.json()["result"])
        time.sleep(5)
    r.raise_for_status()


def oncokb_roles():
    r = requests.get("https://www.oncokb.org/api/v1/utils/cancerGeneList", timeout=120, headers={"Accept": "application/json"})
    r.raise_for_status()
    return {g["hugoSymbol"]: (g.get("geneType") or "") for g in r.json()}


def main():
    d = pd.read_csv(TAB / "differential_promoter_methylation_expression.csv")
    d["expr_sig"] = (d["expr_fdr"] < FDR) & (d["expr_delta"].abs() > LOG2FC)
    d["meth_sig"] = (d["meth_fdr"] < FDR) & (d["meth_delta"].abs() > DELTA_BETA)
    d["category"] = np.select(
        [d["concordant"], d["expr_sig"] & ~d["meth_sig"], d["meth_sig"] & ~d["expr_sig"], d["expr_sig"] & d["meth_sig"]],
        CATS[:4], CATS[4])
    counts = pd.crosstab(d["group"], d["category"]).reindex(GROUPS).reindex(columns=CATS, fill_value=0)
    counts.to_csv(TAB / "expression_layer_counts.csv")
    print(counts)

    roles = oncokb_roles()
    eo = d[d["category"] == "expression only"].copy()
    eo["direction"] = np.where(eo["expr_delta"] > 0, "expression up", "expression down")
    eo["oncokb_gene_type"] = eo["gene"].map(roles).fillna("")
    eo["curated_driver"] = [g in DRIVERS[gr] for g, gr in zip(eo["gene"], eo["group"])]
    eo["nanostring_2012"] = [g in NANOSTRING[gr] for g, gr in zip(eo["gene"], eo["group"])]
    eo = eo.sort_values(["group", "expr_delta"], ascending=[True, False])
    eo[["group", "gene", "direction", "expr_delta", "expr_fdr", "meth_delta", "meth_fdr",
        "oncokb_gene_type", "curated_driver", "nanostring_2012"]].round(4).to_csv(TAB / "expression_only_genes.csv", index=False)

    # ---- NanoString signature check: where does each of the 22 genes land in its own subgroup?
    rows = []
    idx = d.set_index(["group", "gene"])
    for g, gl in NANOSTRING.items():
        for gene in gl:
            if (g, gene) in idx.index:
                r = idx.loc[(g, gene)]
                rows.append({"group": g, "gene": gene, "category": r["category"], "expr_delta": r["expr_delta"],
                             "meth_delta": r["meth_delta"], "expr_fdr": r["expr_fdr"], "meth_fdr": r["meth_fdr"]})
            else:
                rows.append({"group": g, "gene": gene, "category": "not measured", "expr_delta": np.nan,
                             "meth_delta": np.nan, "expr_fdr": np.nan, "meth_fdr": np.nan})
    ns = pd.DataFrame(rows)
    ns.round(4).to_csv(TAB / "nanostring_signature_check.csv", index=False)
    print(ns.to_string(index=False))

    # ---- enrichment: expression-only up, and methylation-only hypomethylated, per subgroup
    bg = pd.read_csv(TAB / "promoter_methylation_vs_expression_all_genes.csv")["gene"].tolist()
    query = {}
    for g in GROUPS:
        up = eo[(eo["group"] == g) & (eo["expr_delta"] > 0)]["gene"].tolist()
        hypo = d[(d["group"] == g) & (d["category"] == "methylation only") & (d["meth_delta"] < 0)]["gene"].tolist()
        if len(up) >= 5:
            query[f"{g} expression only, up"] = up
        if len(hypo) >= 5:
            query[f"{g} methylation only, hypomethylated"] = hypo
    enr = gprofiler(query, bg)
    if len(enr):
        enr = enr[["query", "source", "native", "name", "p_value", "term_size", "query_size",
                   "intersection_size"]].sort_values(["query", "p_value"])
    enr.to_csv(TAB / "expression_layer_enrichment.csv", index=False)
    print(enr.groupby("query").head(3)[["query", "source", "name", "p_value", "intersection_size"]].to_string(index=False) if len(enr) else "no enrichment")

    summary = {
        "counts": {g: counts.loc[g].to_dict() for g in GROUPS},
        "expression_only_up_per_group": {g: int(((eo["group"] == g) & (eo["expr_delta"] > 0)).sum()) for g in GROUPS},
        "expression_only_cancer_genes": {
            g: eo[(eo["group"] == g) & (eo["oncokb_gene_type"] != "")].sort_values("expr_delta", ascending=False)
                [["gene", "direction", "oncokb_gene_type"]].values.tolist() for g in GROUPS},
        "curated_drivers_expression_only": {
            g: eo[(eo["group"] == g) & eo["curated_driver"]][["gene", "direction"]].values.tolist() for g in GROUPS},
        "nanostring_2012_category_counts": ns.groupby("category").size().to_dict(),
        "nanostring_2012_in_own_group": ns[ns["category"].isin(["expression only", "concordant", "both, same direction"])].shape[0],
        "enriched_terms_per_query": enr.groupby("query").size().to_dict() if len(enr) else {},
    }
    json.dump(summary, open(TAB / "expression_layer_summary.json", "w"), indent=2, default=int)
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=1, default=int)[:2500])

    # ---- figure 1: scatter per subgroup with the five categories
    pal = {"concordant": "#c0392b", "expression only": "#e67e22", "methylation only": "#2a78d6",
           "both, same direction": "#8e44ad", "neither": "#d9d9d9"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    for ax, g in zip(axes, GROUPS):
        s = d[d["group"] == g]
        for cat in ["neither", "methylation only", "expression only", "both, same direction", "concordant"]:
            ss = s[s["category"] == cat]
            ax.scatter(ss["meth_delta"], ss["expr_delta"], s=4 if cat == "neither" else 9, color=pal[cat],
                       alpha=0.35 if cat == "neither" else 0.8, linewidths=0, label=f"{cat} ({len(ss)})")
        for gene in NANOSTRING[g]:
            if gene in s["gene"].values:
                r = s[s["gene"] == gene].iloc[0]
                ax.annotate(gene, (r["meth_delta"], r["expr_delta"]), fontsize=7, textcoords="offset points", xytext=(3, 2))
        ax.axhline(0, color="#444", lw=0.6)
        ax.axvline(0, color="#444", lw=0.6)
        ax.set_title(f"{g} vs rest", loc="left", color=GROUP_COLORS[g])
        ax.set_xlabel("delta promoter beta")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("delta log2 expression")
    sns.despine()
    fig.suptitle("Five categories per gene and subgroup; labels: the 22-gene NanoString signature of Northcott et al. 2012",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "expression_layer_scatter.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: top expression-only genes per subgroup
    fig, axes = plt.subplots(1, 4, figsize=(17, 6), sharex=False)
    for ax, g in zip(axes, GROUPS):
        s = eo[eo["group"] == g].nlargest(20, "expr_delta").iloc[::-1]
        colors = ["#8e44ad" if ns_ else ("#c0392b" if r_ else GROUP_COLORS[g]) for ns_, r_ in zip(s["nanostring_2012"], s["oncokb_gene_type"] != "")]
        ax.barh(s["gene"], s["expr_delta"], color=colors, height=0.65)
        ax.set_title(f"{g}: top 20 expression-only genes", loc="left", fontsize=10, color=GROUP_COLORS[g])
        ax.set_xlabel("delta log2 expression vs rest")
        ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    sns.despine()
    fig.suptitle("Expression changes without a promoter methylation change", x=0.01, ha="left", fontsize=10)
    fig.legend(handles=[Patch(color="#8e44ad", label="NanoString 2012 signature gene"),
                        Patch(color="#c0392b", label="OncoKB cancer gene"),
                        Patch(color="#999999", label="other (subgroup colour)")],
               frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "expression_only_top_genes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
