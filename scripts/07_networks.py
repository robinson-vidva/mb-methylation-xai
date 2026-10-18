"""Gene clusters, a co-methylation network, and pathway flow.

Runs after 05_integrate.py. Four parts:
  A  hierarchical clustering of the concordant genes by promoter methylation across all 763 tumours,
     with the matched expression, and g:Profiler enrichment per gene cluster
  B  a co-methylation network of the concordant genes (Spearman rho across tumours), communities,
     hub genes
  C  pathway flow: KEGG membership for Hedgehog, Wnt and phototransduction, and a curated signalling
     order (ligand, receptor, transducer, transcription factor, target) with the change in promoter
     methylation and expression per subgroup for every member
  D  pathway activity per tumour: mean expression of the target genes and mean promoter methylation
     of the same genes, by subgroup

Outputs in results/tables:
  gene_clusters.csv, gene_cluster_enrichment.csv, network_communities.csv, network_hubs.csv,
  pathway_membership_summary.csv, pathway_flow.csv, pathway_activity_scores.csv, networks_summary.json
Figures in results/figures:
  gene_clusters_heatmap.png, comethylation_network.png, pathway_flow.png, pathway_activity.png
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.stats import spearmanr

from common import FIG, GROUPS, GROUP_COLORS, PROC, SEED, TAB

sns.set_theme(style="white", font_scale=0.9)
N_CLUSTERS = 6
EDGE_RHO = 0.6
FDR = 0.05

KEGG = {"Hedgehog": "hsa04340", "Wnt": "hsa04310", "Phototransduction": "hsa04744"}
# Curated signalling order. Only genes with both measurements are drawn.
FLOW = {
    "Hedgehog": [
        ("ligand", ["SHH"]),
        ("co-receptor", ["BOC", "CDON", "GAS1"]),
        ("receptor", ["PTCH1", "PTCH2", "HHIP"]),
        ("transducer", ["SMO", "SUFU", "KIF7"]),
        ("transcription factor", ["GLI1", "GLI2", "GLI3"]),
        ("target", ["MYCN", "CCND1", "CCND2", "SFRP1"]),
    ],
    "Wnt": [
        ("ligand and antagonist", ["WNT1", "WNT3", "DKK1", "DKK2", "WIF1"]),
        ("receptor", ["FZD7", "LRP5", "LRP6", "LGR5"]),
        ("destruction complex", ["APC", "AXIN1", "GSK3B", "CTNNB1"]),
        ("transcription factor", ["TCF7", "TCF7L2", "LEF1"]),
        ("target", ["AXIN2", "NKD1", "SP5", "MYC", "NOTUM"]),
    ],
    "Photoreceptor": [
        ("transcription factor", ["OTX2", "CRX", "NRL", "NR2E3", "RORB"]),
        ("phototransduction", ["RHO", "GNAT1", "GNGT1", "PDE6B", "PDE6H", "SAG", "RCVRN", "GUCA1A"]),
        ("structure and chaperone", ["AIPL1", "IMPG2", "CRB1", "PRPH2", "ROM1"]),
    ],
}
ACTIVITY = {
    "Hedgehog targets": ["GLI1", "PTCH1", "HHIP", "MYCN", "SFRP1"],
    "Wnt targets": ["AXIN2", "NKD1", "LEF1", "DKK1", "SP5", "NOTUM"],
    "Photoreceptor program": ["CRX", "NRL", "RCVRN", "GNAT1", "PDE6H", "AIPL1", "IMPG2", "SAG"],
}


def gprofiler(query_by_name, background):
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {"organism": "hsapiens", "query": query_by_name, "background": background,
               "domain_scope": "custom", "sources": ["GO:BP", "KEGG", "REAC"],
               "user_threshold": 0.05, "significance_threshold_method": "g_SCS", "no_evidences": True}
    for attempt in range(3):
        r = requests.post(url, json=payload, timeout=120)
        if r.ok:
            return pd.DataFrame(r.json()["result"])
        time.sleep(5)
    r.raise_for_status()


def kegg_members():
    """Gene symbols per KEGG pathway, through the KEGG REST API."""
    r = requests.get("https://rest.kegg.jp/list/hsa", timeout=120)
    r.raise_for_status()
    sym = {}
    for line in r.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 4 or ";" not in parts[3]:
            continue
        sym[parts[0]] = parts[3].split(";")[0].split(", ")[0]
    out = {}
    for name, pid in KEGG.items():
        r = requests.get(f"https://rest.kegg.jp/link/hsa/{pid}", timeout=60)
        r.raise_for_status()
        ids = [l.split("\t")[1] for l in r.text.strip().split("\n") if l]
        out[name] = sorted({sym[i] for i in ids if i in sym})
        print(f"KEGG {name} ({pid}): {len(out[name])} genes")
    return out


def main():
    lab = pd.read_csv(PROC / "labels.csv", index_col=0)
    prom = pd.read_parquet(PROC / "promoter_meth.parquet").loc[lab.index]
    prom = prom.rename(columns={"MLL2": "KMT2D", "MLL3": "KMT2C"})
    expr = pd.read_parquet(PROC / "expr.parquet").loc[lab.index]
    genes_all = prom.columns.intersection(expr.columns)
    prom, expr = prom[genes_all], expr[genes_all]
    diff = pd.read_csv(TAB / "differential_promoter_methylation_expression.csv")
    conc = pd.read_csv(TAB / "concordant_genes_per_group.csv")
    cgenes = [g for g in pd.unique(conc["gene"]) if g in genes_all]
    # subgroup in which each concordant gene is concordant (first if several), and its direction
    gene_group = conc.drop_duplicates("gene").set_index("gene")
    order = lab.sort_values(["group", "subtype"], key=lambda s: s.map({g: i for i, g in enumerate(GROUPS)}) if s.name == "group" else s).index
    summary = {"concordant_genes_used": len(cgenes)}

    # ---- A. clustering of concordant genes by promoter methylation
    M = prom[cgenes]
    Mc = M - M.mean(axis=0)
    Z = linkage(Mc.T.values, method="average", metric="correlation")
    clusters = fcluster(Z, N_CLUSTERS, criterion="maxclust")
    leaf_order = leaves_list(Z)
    genes_sorted = [cgenes[i] for i in leaf_order]
    cl = pd.DataFrame({"gene": cgenes, "cluster": clusters}).set_index("gene")
    for g in GROUPS:
        m = (lab["group"] == g).values
        cl[f"delta_beta_{g}"] = (M.values[m].mean(0) - M.values[~m].mean(0))
        cl[f"delta_expr_{g}"] = (expr[cgenes].values[m].mean(0) - expr[cgenes].values[~m].mean(0))
    cl["concordant_in"] = gene_group.loc[cl.index, "group"].values
    cl["hypomethylated_in"] = cl[[f"delta_beta_{g}" for g in GROUPS]].idxmin(axis=1).str.replace("delta_beta_", "")
    cl = cl.loc[genes_sorted]
    cl.round(4).to_csv(TAB / "gene_clusters.csv")
    cluster_sizes = cl.groupby("cluster").size().to_dict()
    cluster_group = cl.groupby("cluster")["hypomethylated_in"].agg(lambda s: s.value_counts().index[0]).to_dict()
    summary["cluster_sizes"] = {int(k): int(v) for k, v in cluster_sizes.items()}
    summary["cluster_dominant_hypomethylated_group"] = {int(k): v for k, v in cluster_group.items()}
    print(cl.groupby("cluster")["hypomethylated_in"].value_counts().unstack(fill_value=0))

    bg = pd.read_csv(TAB / "promoter_methylation_vs_expression_all_genes.csv")["gene"].tolist()
    query = {f"cluster {k}": cl.index[cl["cluster"] == k].tolist() for k in sorted(cluster_sizes) if cluster_sizes[k] >= 5}
    enr = gprofiler(query, bg)
    if len(enr):
        enr = enr[["query", "source", "native", "name", "p_value", "term_size", "query_size",
                   "intersection_size"]].sort_values(["query", "p_value"])
    enr.to_csv(TAB / "gene_cluster_enrichment.csv", index=False)
    summary["enriched_terms_per_cluster"] = enr.groupby("query").size().to_dict() if len(enr) else {}

    # heatmap figure: methylation (centred beta) and expression (z-score), genes in cluster order
    E = expr[cgenes]
    Ez = (E - E.mean(axis=0)) / E.std(axis=0).replace(0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"wspace": 0.08})
    for ax, mat, title, vmax, cmap in [
        (axes[0], Mc.loc[order, genes_sorted].T, "promoter methylation (beta minus gene mean)", 0.4, "RdBu_r"),
        (axes[1], Ez.loc[order, genes_sorted].T, "expression (z-score per gene)", 2.5, "RdBu_r")]:
        ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=10, pad=34)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("763 tumours, ordered by subgroup and subtype")
        # subgroup colour bar on top
        grp = lab.loc[order, "group"].values
        start = 0
        for i in range(1, len(grp) + 1):
            if i == len(grp) or grp[i] != grp[start]:
                ax.add_patch(plt.Rectangle((start - 0.5, -0.03 * len(genes_sorted) - 0.5), i - start, 0.025 * len(genes_sorted),
                                           color=GROUP_COLORS[grp[start]], clip_on=False))
                ax.text((start + i) / 2, -0.045 * len(genes_sorted), grp[start], ha="center", va="bottom", fontsize=8)
                start = i
    # cluster bar on the left of the first panel
    cvals = cl["cluster"].values
    pal = sns.color_palette("Set2", N_CLUSTERS)
    start = 0
    for i in range(1, len(cvals) + 1):
        if i == len(cvals) or cvals[i] != cvals[start]:
            axes[0].add_patch(plt.Rectangle((-0.02 * len(order) - 0.5, start - 0.5), 0.015 * len(order), i - start,
                                            color=pal[cvals[start] - 1], clip_on=False))
            axes[0].text(-0.025 * len(order), (start + i) / 2, f"c{cvals[start]}", ha="right", va="center", fontsize=8)
            start = i
    axes[0].set_ylabel(f"{len(genes_sorted)} concordant genes, hierarchical clustering order (c1 to c{N_CLUSTERS})", labelpad=28)
    fig.suptitle("Concordant genes across all tumours: promoter methylation and expression, same gene order",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "gene_clusters_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- B. co-methylation network
    rho = spearmanr(M.values).statistic
    np.fill_diagonal(rho, 0)
    G = nx.Graph()
    G.add_nodes_from(cgenes)
    iu = np.triu_indices(len(cgenes), 1)
    for i, j in zip(*iu):
        if abs(rho[i, j]) >= EDGE_RHO:
            G.add_edge(cgenes[i], cgenes[j], weight=abs(rho[i, j]), sign=int(np.sign(rho[i, j])))
    G.remove_nodes_from([n for n in cgenes if G.degree(n) == 0])
    comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    comm_of = {n: k + 1 for k, c in enumerate(comms) for n in c}
    net = pd.DataFrame({"gene": list(G.nodes), "community": [comm_of[n] for n in G.nodes],
                        "degree": [G.degree(n) for n in G.nodes],
                        "concordant_in": [gene_group.loc[n, "group"] for n in G.nodes],
                        "hypomethylated_in": [cl.loc[n, "hypomethylated_in"] for n in G.nodes],
                        "cluster_A": [int(cl.loc[n, "cluster"]) for n in G.nodes]}).sort_values(["community", "degree"], ascending=[True, False])
    net.to_csv(TAB / "network_communities.csv", index=False)
    hubs = net.groupby("community").head(5)
    hubs.to_csv(TAB / "network_hubs.csv", index=False)
    summary["network"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                          "negative_edges": int(sum(1 for _, _, d in G.edges(data=True) if d["sign"] < 0)),
                          "communities_min5": int(sum(1 for c in comms if len(c) >= 5)),
                          "dominant_group_per_community": {int(k): v for k, v in
                              net[net["community"].isin([c for c in net["community"].unique() if (net["community"] == c).sum() >= 5])]
                              .groupby("community")["hypomethylated_in"].agg(lambda s: s.value_counts().index[0]).to_dict().items()}}
    print(json.dumps(summary["network"], indent=1))

    pos = nx.spring_layout(G, seed=SEED, k=0.35, weight="weight")
    fig, ax = plt.subplots(figsize=(11, 9))
    pos_e = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    neg_e = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]
    nx.draw_networkx_edges(G, pos, edgelist=pos_e, alpha=0.15, width=0.6, edge_color="#777", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_e, alpha=0.4, width=0.8, edge_color="#c0392b", style="dashed", ax=ax)
    deg = dict(G.degree)
    nx.draw_networkx_nodes(G, pos, node_size=[20 + 6 * deg[n] for n in G.nodes],
                           node_color=[GROUP_COLORS[cl.loc[n, "hypomethylated_in"]] for n in G.nodes],
                           edgecolors="white", linewidths=0.5, ax=ax)
    labels = {n: n for n in hubs["gene"]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    for g in GROUPS:
        ax.scatter([], [], color=GROUP_COLORS[g], label=f"hypomethylated in {g}")
    ax.plot([], [], color="#777", lw=1, label=f"co-methylation rho >= {EDGE_RHO}")
    ax.plot([], [], color="#c0392b", lw=1, ls="--", label=f"anti-correlated rho <= -{EDGE_RHO}")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"Co-methylation network of {G.number_of_nodes()} concordant genes, {G.number_of_edges()} edges, "
                 f"{len([c for c in comms if len(c) >= 5])} communities of 5 or more genes (hubs labelled)", loc="left", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "comethylation_network.png", dpi=150)
    plt.close(fig)

    # ---- C. pathway flow
    members = kegg_members()
    dsum = []
    d_idx = diff.set_index(["group", "gene"])
    for name, glist in members.items():
        present = [g for g in glist if g in genes_all]
        for g in GROUPS:
            d = d_idx.loc[g].reindex(present)
            dsum.append({"pathway": name, "kegg": KEGG[name], "members": len(glist), "with_both_measurements": len(present),
                         "group": g,
                         "expression_up": int(((d["expr_fdr"] < FDR) & (d["expr_delta"] > 1)).sum()),
                         "expression_down": int(((d["expr_fdr"] < FDR) & (d["expr_delta"] < -1)).sum()),
                         "promoter_hypomethylated": int(((d["meth_fdr"] < FDR) & (d["meth_delta"] < -0.1)).sum()),
                         "promoter_hypermethylated": int(((d["meth_fdr"] < FDR) & (d["meth_delta"] > 0.1)).sum()),
                         "concordant": int(d["concordant"].fillna(False).astype(bool).sum())})
    dsum = pd.DataFrame(dsum)
    dsum.to_csv(TAB / "pathway_membership_summary.csv", index=False)
    print(dsum.pivot(index=["pathway", "with_both_measurements"], columns="group", values=["expression_up", "concordant"]))

    rows = []
    for name, stages in FLOW.items():
        for stage, glist in stages:
            for g in glist:
                if g not in genes_all:
                    continue
                for grp in GROUPS:
                    d = d_idx.loc[(grp, g)]
                    rows.append({"pathway": name, "stage": stage, "gene": g, "group": grp,
                                 "delta_promoter_beta": d["meth_delta"], "meth_fdr": d["meth_fdr"],
                                 "delta_log2_expression": d["expr_delta"], "expr_fdr": d["expr_fdr"],
                                 "concordant": bool(d["concordant"])})
    flow = pd.DataFrame(rows)
    flow.round(4).to_csv(TAB / "pathway_flow.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.55})
    for ax, name in zip(axes, FLOW):
        f = flow[flow["pathway"] == name]
        genes_in = list(dict.fromkeys(f["gene"]))
        stage_of = dict(zip(f["gene"], f["stage"]))
        meth = f.pivot(index="gene", columns="group", values="delta_promoter_beta").loc[genes_in, GROUPS]
        ex = f.pivot(index="gene", columns="group", values="delta_log2_expression").loc[genes_in, GROUPS]
        mf = f.pivot(index="gene", columns="group", values="meth_fdr").loc[genes_in, GROUPS]
        ef = f.pivot(index="gene", columns="group", values="expr_fdr").loc[genes_in, GROUPS]
        mat = np.hstack([meth.values / 0.25, ex.values / 2.5])  # scale both to about -1..1
        ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
        for i, g in enumerate(genes_in):
            for j, grp in enumerate(GROUPS):
                if mf.iloc[i, j] < FDR and abs(meth.iloc[i, j]) > 0.1:
                    ax.text(j, i, f"{meth.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(mat[i, j]) > 0.6 else "#111")
                if ef.iloc[i, j] < FDR and abs(ex.iloc[i, j]) > 1:
                    ax.text(4 + j, i, f"{ex.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(mat[i, 4 + j]) > 0.6 else "#111")
                if bool(f[(f["gene"] == g) & (f["group"] == grp)]["concordant"].iloc[0]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111", lw=1.2))
                    ax.add_patch(plt.Rectangle((4 + j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111", lw=1.2))
        ax.axvline(3.5, color="#222", lw=1)
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"{g}" for g in GROUPS] * 2, rotation=60, ha="right", fontsize=8)
        ax.text(1.5, -0.9, "delta promoter beta", ha="center", va="bottom", fontsize=8)
        ax.text(5.5, -0.9, "delta log2 expression", ha="center", va="bottom", fontsize=8)
        ax.set_yticks(range(len(genes_in)))
        ax.set_yticklabels(genes_in, fontsize=8)
        # stage separators and labels
        prev = None
        for i, g in enumerate(genes_in):
            if stage_of[g] != prev:
                if i > 0:
                    ax.axhline(i - 0.5, color="#222", lw=0.8)
                ax.text(-1.6, i - 0.3, stage_of[g], ha="right", va="top", fontsize=7, style="italic", color="#444")
                prev = stage_of[g]
        ax.set_title(f"{name}", loc="left", fontsize=11, pad=26)
        ax.tick_params(length=0)
    fig.suptitle("Pathway flow: change of promoter methylation and expression per subgroup along the signalling order "
                 "(numbers: FDR < 0.05 and above threshold; boxed: concordant)", x=0.01, ha="left", fontsize=10)
    fig.savefig(FIG / "pathway_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- D. pathway activity per tumour
    act = pd.DataFrame(index=lab.index)
    for name, glist in ACTIVITY.items():
        gl = [g for g in glist if g in genes_all]
        Ez_ = (expr[gl] - expr[gl].mean()) / expr[gl].std().replace(0, 1)
        act[f"{name} | expression score"] = Ez_.mean(axis=1)
        act[f"{name} | promoter beta"] = prom[gl].mean(axis=1)
        summary[f"activity_genes_{name}"] = gl
    act = act.join(lab[["group", "subtype"]])
    act.index.name = "sample"
    act.round(4).to_csv(TAB / "pathway_activity_scores.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, name in enumerate(ACTIVITY):
        for i, kind in enumerate(["expression score", "promoter beta"]):
            ax = axes[i, j]
            col = f"{name} | {kind}"
            sns.boxplot(data=act, x="group", y=col, order=GROUPS, hue="group", palette=GROUP_COLORS,
                        legend=False, ax=ax, fliersize=1.5, width=0.6)
            ax.set_xlabel("")
            ax.set_ylabel("mean z-scored expression" if i == 0 else "mean promoter beta")
            ax.set_title(f"{name}: {kind}", loc="left", fontsize=9)
    sns.despine()
    fig.suptitle("Pathway activity per tumour: target gene expression (top) and promoter methylation of the same genes (bottom)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pathway_activity.png", dpi=150)
    plt.close(fig)
    med = act.groupby("group")[[c for c in act.columns if "|" in c]].median().round(3)
    summary["activity_median_by_group"] = {c: med[c].to_dict() for c in med.columns}
    print(med.T)

    json.dump(summary, open(TAB / "networks_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
