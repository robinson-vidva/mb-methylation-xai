"""Gene-level annotation: names, cancer roles, literature and pathways.

For every gene that appears in the analysis (concordant genes, curated drivers, pathway flow genes,
network hubs, genes under the top attributed CpGs) this script collects:
  - full gene name and the NCBI summary paragraph (MyGene.info)
  - oncogene or tumour suppressor status (OncoKB cancer gene list, public; and the COSMIC
    Cancer Gene Census "role in cancer" as carried by the Open Targets Platform)
  - cancer hallmarks (COSMIC, through Open Targets) and Reactome pathway membership
  - PubMed counts: papers mentioning the gene with medulloblastoma, and with cancer in general,
    plus the three most relevant medulloblastoma PMIDs with titles
Then, for every concordant gene and subgroup, it reads the methylation change against the role:
an oncogene switched on by promoter hypomethylation and a tumour suppressor silenced by promoter
hypermethylation are the two readings that fit a causal story; the other two combinations do not.

Outputs in results/tables:
  gene_annotation.csv              one row per gene
  gene_regulation_concordant.csv   one row per concordant gene and subgroup, with the reading
  gene_literature_top.csv          top medulloblastoma papers per gene (PMID, year, title)
  gene_annotation_summary.json
Figures: gene_roles_by_group.png, signalling_map.png
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
OT = "https://api.platform.opentargets.org/api/v4/graphql"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DRIVERS = {
    "WNT": ["CTNNB1", "DDX3X", "SMARCA4", "TP53", "KMT2D"],
    "SHH": ["PTCH1", "SUFU", "SMO", "GLI2", "MYCN", "TP53", "TERT", "KMT2D", "DDX3X"],
    "Group3": ["MYC", "GFI1", "GFI1B", "OTX2", "SMARCA4", "KBTBD4"],
    "Group4": ["KDM6A", "PRDM6", "SNCAIP", "ZMYM3", "KBTBD4", "MYCN", "OTX2", "CDK6"],
}
READINGS = ["oncogene switched on", "tumour suppressor silenced", "oncogene silenced",
            "tumour suppressor switched on", "dual role gene changed", "no cancer role annotated"]


def gql(query, variables=None, tries=4):
    for attempt in range(tries):
        r = requests.post(OT, json={"query": query, "variables": variables or {}}, timeout=120)
        if r.ok and "errors" not in r.json():
            return r.json()["data"]
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Open Targets query failed: {r.status_code} {r.text[:300]}")


def mygene(symbols):
    out = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        r = requests.post("https://mygene.info/v3/query", timeout=120,
                          data={"q": ",".join(chunk), "scopes": "symbol", "species": "human",
                                "fields": "name,summary,entrezgene,ensembl.gene"})
        r.raise_for_status()
        for h in r.json():
            if h.get("notfound") or h["query"] in out:
                continue
            ens = h.get("ensembl")
            if isinstance(ens, list):
                ens = ens[0]
            out[h["query"]] = {"gene_name": h.get("name", ""), "ncbi_summary": (h.get("summary") or ""),
                               "entrez_id": h.get("entrezgene", ""), "ensembl_id": (ens or {}).get("gene", "")}
    print(f"MyGene: {len(out)}/{len(symbols)} symbols")
    return out


def oncokb():
    r = requests.get("https://www.oncokb.org/api/v1/utils/cancerGeneList", timeout=120, headers={"Accept": "application/json"})
    r.raise_for_status()
    out = {}
    for g in r.json():
        out[g["hugoSymbol"]] = {"oncokb_gene_type": g.get("geneType", ""), "oncokb_sources": int(g.get("occurrenceCount") or 0),
                                "in_cancer_gene_census": bool(g.get("sangerCGC")), "vogelstein": bool(g.get("vogelstein"))}
    print(f"OncoKB cancer gene list: {len(out)} genes")
    return out


def open_targets(ensembl_ids):
    out = {}
    ids = [e for e in ensembl_ids if e]
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        parts = [f't{j}: target(ensemblId: "{e}") {{ id approvedSymbol approvedName '
                 f'hallmarks {{ attributes {{ name description }} cancerHallmarks {{ label impact }} }} '
                 f'pathways {{ pathway pathwayId topLevelTerm }} }}' for j, e in enumerate(chunk)]
        data = gql("query { " + " ".join(parts) + " }")
        for j, e in enumerate(chunk):
            t = data[f"t{j}"]
            if t is None:
                continue
            h = t["hallmarks"] or {}
            roles = [a["description"] for a in h.get("attributes", []) if a["name"] == "role in cancer"]
            hm = {}
            for x in h.get("cancerHallmarks", []):
                if x["impact"]:
                    hm.setdefault(x["label"], set()).add(x["impact"])
            paths = t["pathways"] or []
            out[e] = {"ot_name": t["approvedName"] or "",
                      "cgc_role_in_cancer": "; ".join(sorted(set(roles))),
                      "cancer_hallmarks": "; ".join(f"{k} ({'/'.join(sorted(v))})" for k, v in sorted(hm.items())),
                      "reactome_top_level": "; ".join(sorted({p["topLevelTerm"] for p in paths})),
                      "reactome_pathways": "; ".join(sorted({p["pathway"] for p in paths}))[:600],
                      "n_reactome_pathways": len(paths)}
        print(f"Open Targets: {min(i + 20, len(ids))}/{len(ids)}")
    return out


def pubmed_count(term):
    for attempt in range(4):
        r = requests.get(EUTILS + "esearch.fcgi", timeout=60,
                         params={"db": "pubmed", "term": term, "retmode": "json", "retmax": 3, "sort": "relevance"})
        if r.ok:
            try:
                j = r.json()["esearchresult"]
                return int(j["count"]), j["idlist"]
            except (KeyError, ValueError):
                pass
        time.sleep(2 * (attempt + 1))
    return np.nan, []


def pubmed_titles(pmids):
    out = {}
    pmids = sorted(set(pmids))
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        r = requests.post(EUTILS + "esummary.fcgi", timeout=120, data={"db": "pubmed", "id": ",".join(chunk), "retmode": "json"})
        if not r.ok:
            time.sleep(3)
            continue
        res = r.json().get("result", {})
        for p in chunk:
            d = res.get(p, {})
            out[p] = (d.get("pubdate", "")[:4], d.get("title", ""), d.get("fulljournalname", ""))
        time.sleep(0.4)
    return out


def role_of(row):
    """Combine OncoKB gene type and the CGC role into oncogene, tumour suppressor, both, or none."""
    t = (row.get("oncokb_gene_type") or "").upper()
    c = (row.get("cgc_role_in_cancer") or "").lower()
    onc = "ONCOGENE" in t or "oncogene" in c
    tsg = "TSG" in t or "tsg" in c
    if onc and tsg:
        return "both"
    if onc:
        return "oncogene"
    if tsg:
        return "tumour suppressor"
    return "none"


def reading(direction, role):
    on = "expression up" in direction
    if role == "none":
        return READINGS[5]
    if role == "both":
        return READINGS[4]
    if role == "oncogene":
        return READINGS[0] if on else READINGS[2]
    return READINGS[3] if on else READINGS[1]


def main():
    conc = pd.read_csv(TAB / "concordant_genes_per_group.csv")
    conc["direction"] = np.where(conc["meth_delta"] < 0, "promoter hypomethylated, expression up",
                                 "promoter hypermethylated, expression down")
    top = pd.read_csv(TAB / "top_cpgs_per_group.csv")
    flow = pd.read_csv(TAB / "pathway_flow.csv")
    hubs = pd.read_csv(TAB / "network_hubs.csv")
    sets = {
        "concordant": set(conc["gene"]),
        "driver": {g for l in DRIVERS.values() for g in l},
        "pathway_flow": set(flow["gene"]),
        "network_hub": set(hubs["gene"]),
        "top_attributed_cpg": set(top["gene"].dropna()),
    }
    symbols = sorted(set().union(*sets.values()))
    symbols = [s for s in symbols if s and s == s]
    print("genes to annotate:", len(symbols))

    mg = mygene(symbols)
    ok = oncokb()
    ot = open_targets([mg[s]["ensembl_id"] for s in symbols if s in mg])
    ther = pd.read_csv(TAB / "therapy_concordant_genes.csv").drop_duplicates("gene").set_index("gene")
    ther_d = pd.read_csv(TAB / "therapy_driver_genes.csv").drop_duplicates("gene").set_index("gene")
    mb_assoc = pd.concat([ther["mb_association_score"], ther_d["mb_association_score"]])
    mb_assoc = mb_assoc[~mb_assoc.index.duplicated()]

    rows, top_pmids = [], {}
    for k, s in enumerate(symbols):
        m = mg.get(s, {})
        o = ok.get(s, {})
        t = ot.get(m.get("ensembl_id", ""), {})
        n_mb, ids_mb = pubmed_count(f"{s}[Title/Abstract] AND medulloblastoma[Title/Abstract]")
        time.sleep(0.35)
        n_ca, _ = pubmed_count(f"{s}[Title/Abstract] AND (cancer[Title/Abstract] OR tumor[Title/Abstract] OR tumour[Title/Abstract])")
        time.sleep(0.35)
        top_pmids[s] = ids_mb
        row = {"gene": s, "gene_name": m.get("gene_name") or t.get("ot_name", ""), "entrez_id": m.get("entrez_id", ""),
               "ensembl_id": m.get("ensembl_id", ""),
               "in_sets": "; ".join(k2 for k2, v in sets.items() if s in v),
               "oncokb_gene_type": o.get("oncokb_gene_type", ""), "oncokb_sources": o.get("oncokb_sources", 0),
               "in_cancer_gene_census": o.get("in_cancer_gene_census", False),
               "cgc_role_in_cancer": t.get("cgc_role_in_cancer", ""), "cancer_hallmarks": t.get("cancer_hallmarks", ""),
               "reactome_top_level": t.get("reactome_top_level", ""), "reactome_pathways": t.get("reactome_pathways", ""),
               "pubmed_medulloblastoma": n_mb, "pubmed_cancer": n_ca,
               "mb_association_score": float(mb_assoc.get(s, np.nan)),
               "ncbi_summary": (m.get("ncbi_summary") or "")[:700]}
        row["cancer_role"] = role_of(row)
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"PubMed: {k + 1}/{len(symbols)}")
    ann = pd.DataFrame(rows).set_index("gene")
    ann.to_csv(TAB / "gene_annotation.csv")

    titles = pubmed_titles([p for ids in top_pmids.values() for p in ids])
    lit = pd.DataFrame([{"gene": s, "rank": r + 1, "pmid": p, "year": titles.get(p, ("", "", ""))[0],
                         "title": titles.get(p, ("", "", ""))[1], "journal": titles.get(p, ("", "", ""))[2]}
                        for s, ids in top_pmids.items() for r, p in enumerate(ids)])
    lit.to_csv(TAB / "gene_literature_top.csv", index=False)

    # ---- regulation reading for every concordant gene and subgroup
    reg = conc[["group", "gene", "direction", "meth_delta", "expr_delta", "meth_fdr", "expr_fdr"]].copy()
    for c in ["gene_name", "cancer_role", "oncokb_gene_type", "cgc_role_in_cancer", "cancer_hallmarks",
              "reactome_top_level", "pubmed_medulloblastoma", "pubmed_cancer", "mb_association_score"]:
        reg[c] = reg["gene"].map(ann[c])
    reg["reading"] = [reading(d, r) for d, r in zip(reg["direction"], reg["cancer_role"])]
    reg = reg.sort_values(["group", "reading", "pubmed_medulloblastoma"], ascending=[True, True, False])
    reg.round(4).to_csv(TAB / "gene_regulation_concordant.csv", index=False)

    summary = {
        "genes_annotated": int(len(ann)),
        "with_cancer_role": int((ann["cancer_role"] != "none").sum()),
        "cancer_role_counts": ann["cancer_role"].value_counts().to_dict(),
        "readings_per_group": {g: reg[reg["group"] == g]["reading"].value_counts().to_dict() for g in GROUPS},
        "fits_causal_story_per_group": {
            g: sorted(reg[(reg["group"] == g) & reg["reading"].isin(READINGS[:2])]["gene"].tolist()) for g in GROUPS},
        "against_causal_story_per_group": {
            g: sorted(reg[(reg["group"] == g) & reg["reading"].isin(READINGS[2:4])]["gene"].tolist()) for g in GROUPS},
        "most_cited_in_medulloblastoma_per_group": {
            g: reg[reg["group"] == g].nlargest(8, "pubmed_medulloblastoma")[["gene", "pubmed_medulloblastoma"]].values.tolist()
            for g in GROUPS},
        "concordant_genes_never_mentioned_with_medulloblastoma": int((reg.drop_duplicates("gene")["pubmed_medulloblastoma"] == 0).sum()),
    }
    json.dump(summary, open(TAB / "gene_annotation_summary.json", "w"), indent=2, default=int)
    print(json.dumps(summary, indent=1, default=int)[:3000])

    # ---- figure 1: readings per subgroup
    pal = {READINGS[0]: "#c0392b", READINGS[1]: "#2a78d6", READINGS[2]: "#f1a9a0", READINGS[3]: "#a9cce3",
           READINGS[4]: "#8e44ad", READINGS[5]: "#e5e5e5"}
    ct = reg.groupby(["group", "reading"]).size().unstack(fill_value=0).reindex(GROUPS).reindex(columns=READINGS, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    left = np.zeros(len(GROUPS))
    for r_ in READINGS:
        ax.barh(GROUPS, ct[r_].values, left=left, color=pal[r_], edgecolor="white", label=r_)
        left += ct[r_].values
    ax.invert_yaxis()
    ax.set_xlabel("concordant genes")
    ax.set_title("How the methylation change reads against the gene's cancer role (OncoKB, COSMIC)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "gene_roles_by_group.png", dpi=160)
    plt.close(fig)

    plot_signalling_map(reg)
    print(reg[reg["reading"].isin(READINGS[:4])][["group", "gene", "gene_name", "cancer_role", "direction", "pubmed_medulloblastoma", "reading"]].to_string(index=False))


def plot_signalling_map(reg):
    """Figure 2: concordant genes with a cancer role or a Signal Transduction pathway, by subgroup."""
    sel = reg[(reg["cancer_role"] != "none") | reg["reactome_top_level"].fillna("").str.contains("Signal Transduction")].copy()
    sel["size"] = 20 + 12 * np.log1p(sel["pubmed_medulloblastoma"].fillna(0))
    fig, axes = plt.subplots(1, 4, figsize=(17, 7.5), sharey=False)
    for ax, g in zip(axes, GROUPS):
        d = sel[sel["group"] == g].sort_values(["direction", "pubmed_medulloblastoma"], ascending=[False, True])
        y = np.arange(len(d))
        colors = ["#c0392b" if "up" in x else "#2a78d6" for x in d["direction"]]
        ax.scatter(d["expr_delta"], y, s=d["size"], c=colors, edgecolor="white", linewidth=0.5, zorder=3)
        for yi, (gname, role, n) in enumerate(zip(d["gene"], d["cancer_role"], d["pubmed_medulloblastoma"])):
            tag = {"oncogene": " [ONC]", "tumour suppressor": " [TSG]", "both": " [ONC/TSG]"}.get(role, "")
            ax.text(-0.2 if d["expr_delta"].iloc[yi] > 0 else 0.2, yi, f"{gname}{tag}",
                    ha="right" if d["expr_delta"].iloc[yi] > 0 else "left", va="center", fontsize=7)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_yticks([])
        ax.set_xlabel("delta log2 expression vs other subgroups")
        ax.set_title(f"{g}: {len(d)} genes", loc="left", color=GROUP_COLORS[g], fontsize=11)
        lim = max(1.5, float(d["expr_delta"].abs().max()) + 1.5) if len(d) else 3
        ax.set_xlim(-lim, lim)
    h1 = axes[0].scatter([], [], c="#c0392b", label="promoter hypomethylated, expression up")
    h2 = axes[0].scatter([], [], c="#2a78d6", label="promoter hypermethylated, expression down")
    sns.despine(left=True)
    fig.suptitle("Concordant genes with a cancer role (ONC, TSG) or in a Reactome signal transduction pathway; "
                 "dot size = medulloblastoma papers on PubMed", x=0.01, ha="left", fontsize=10)
    fig.legend(handles=[h1, h2], frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "signalling_map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
