"""Explain the group classifier: which CpGs drive each prediction, and which genes they sit in.

Method: integrated gradients (Captum) on the trained MLP, baseline = training mean profile,
attributions averaged over the correctly classified training samples of each group.
Also: autoencoder latent space and UMAP figure (representation learning).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from captum.attr import IntegratedGradients
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from common import FIG, GROUPS, GROUP_COLORS, PROC, SEED, SUBTYPE_MARKERS, TAB, load_manifest
from models import MLP, encode, predict_proba, train_autoencoder

sns.set_theme(style="white", font_scale=0.9)
TOP_N = 50


def main():
    X = pd.read_parquet(PROC / "beta_top.parquet")
    lab = pd.read_csv(PROC / "labels.csv", index_col=0).loc[X.index]
    tr = (lab["split"] == "train").values
    enc = {c: i for i, c in enumerate(GROUPS)}
    y = lab["group"].map(enc).values

    model = MLP(X.shape[1], len(GROUPS))
    model.load_state_dict(torch.load(PROC / "mlp_group.pt"))
    model.eval()

    # ---- integrated gradients per group
    man = load_manifest().reindex(X.columns)
    baseline = torch.as_tensor(X.values[tr].mean(axis=0, keepdims=True), dtype=torch.float32)
    ig = IntegratedGradients(model)
    pred_tr = predict_proba(model, X.values[tr]).argmax(1)
    attr_rows, top_rows = [], []
    for g, gi in enc.items():
        sel = tr.copy()
        sel[tr] = (y[tr] == gi) & (pred_tr == gi)
        xs = torch.as_tensor(X.values[sel], dtype=torch.float32)
        attr = ig.attribute(xs, baselines=baseline, target=gi, n_steps=32).numpy()
        mean_attr = attr.mean(axis=0)
        delta_beta = X.values[sel].mean(axis=0) - X.values[tr & ~sel].mean(axis=0)
        df = pd.DataFrame({"cpg": X.columns, "group": g, "mean_attribution": mean_attr,
                           "abs_attribution": np.abs(mean_attr),
                           "delta_beta_vs_rest": delta_beta,
                           "chr": man["CHR"].values, "pos": man["MAPINFO"].values,
                           "gene": man["UCSC_RefGene_Name"].fillna("").str.split(";").str[0].values,
                           "region": man["UCSC_RefGene_Group"].fillna("").str.split(";").str[0].values,
                           "cpg_island": man["Relation_to_UCSC_CpG_Island"].fillna("").values})
        attr_rows.append(df)
        top_rows.append(df.sort_values("abs_attribution", ascending=False).head(TOP_N))
    attr_all = pd.concat(attr_rows)
    attr_all.to_parquet(PROC / "attributions_group.parquet")
    top = pd.concat(top_rows)
    top["direction"] = np.where(top["delta_beta_vs_rest"] > 0, "hypermethylated", "hypomethylated")
    top.round(5).to_csv(TAB / "top_cpgs_per_group.csv", index=False)

    # ---- figure: top 15 CpGs per group, bars by attribution, coloured by group
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=False)
    for ax, g in zip(axes, GROUPS):
        d = top[top["group"] == g].head(15).iloc[::-1]
        labels = [f"{c} ({gn})" if gn else c for c, gn in zip(d["cpg"], d["gene"])]
        ax.barh(labels, d["mean_attribution"], color=GROUP_COLORS[g], height=0.6)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_title(g, loc="left")
        ax.set_xlabel("mean integrated gradient")
        ax.tick_params(axis="y", labelsize=7)
    sns.despine()
    fig.suptitle("CpGs with the largest attribution per subgroup (MLP, training samples)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "top_cpgs_per_group.png", dpi=160)
    plt.close(fig)

    # ---- autoencoder latent space, UMAP, classifier on latent
    ae = train_autoencoder(X.values[tr], latent=16)
    Z = encode(ae, X.values)
    np.save(PROC / "latent.npy", Z)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for task in ["group", "subtype"]:
        yt = pd.factorize(lab[task])[0]
        for name, feats in [("raw 10,000 CpGs", X.values), ("autoencoder latent (16 dims)", Z)]:
            sc = cross_val_score(LogisticRegression(max_iter=3000, C=0.5, random_state=SEED),
                                 feats[tr], yt[tr], cv=skf, scoring="balanced_accuracy")
            rows.append({"task": task, "features": name, "cv_balanced_accuracy_mean": sc.mean(),
                         "cv_balanced_accuracy_std": sc.std()})
    pd.DataFrame(rows).round(3).to_csv(TAB / "latent_vs_raw.csv", index=False)

    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7, 6))
    for g in GROUPS:
        for st, mk in SUBTYPE_MARKERS.items():
            m = ((lab["group"] == g) & (lab["subtype"].str.endswith(st))).values
            if m.sum() == 0:
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=22, marker=mk, color=GROUP_COLORS[g],
                       edgecolor="white", linewidth=0.4, label=f"{g} {st}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Autoencoder latent space of 763 medulloblastomas (colour = group, marker = subtype)", loc="left", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, ncol=2, markerscale=1.2)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(FIG / "umap_latent.png", dpi=160)
    plt.close(fig)
    print(pd.read_csv(TAB / "latent_vs_raw.csv"))
    print(top.groupby("group").head(5)[["group", "cpg", "gene", "region", "direction", "mean_attribution"]])


if __name__ == "__main__":
    main()
