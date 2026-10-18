"""From concordant genes to therapy hypotheses, using the Open Targets Platform.

For every concordant gene per group (promoter methylation and expression changed in opposite
directions, from 05_integrate.py) and for the curated driver genes, this script fetches from
Open Targets (https://platform.opentargets.org, public GraphQL API, no login):
  - tractability: is there an approved drug, a clinical-stage compound, or a druggable pocket
    for a small molecule (SM) or an antibody (AB)
  - drugs and clinical candidates against the target, with their highest clinical stage
  - the Open Targets association score between the target and medulloblastoma (MONDO_0007959)
  - whether the gene is a common-essential gene in DepMap screens

Outputs in results/tables:
  therapy_concordant_genes.csv   one row per concordant gene and group, with all annotations
  therapy_driver_genes.csv       the same annotations for the curated driver genes
  therapy_candidates.csv         concordant genes with an approved or clinical-stage compound
  therapy_summary.json           counts per group
Figure: results/figures/therapy_druggability.png

This is hypothesis generation. A gene that is switched on in a subgroup and has a drug against it
is a lead to read about, not a treatment recommendation.
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
URL = "https://api.platform.opentargets.org/api/v4/graphql"
MB_EFO = "MONDO_0007959"
DRIVERS = {
    "WNT": ["CTNNB1", "DDX3X", "SMARCA4", "TP53", "KMT2D"],
    "SHH": ["PTCH1", "SUFU", "SMO", "GLI2", "MYCN", "TP53", "TERT", "KMT2D", "DDX3X"],
    "Group3": ["MYC", "GFI1", "GFI1B", "OTX2", "SMARCA4", "KBTBD4"],
    "Group4": ["KDM6A", "PRDM6", "SNCAIP", "ZMYM3", "KBTBD4", "MYCN", "OTX2", "CDK6"],
}
TIERS = ["approved drug", "clinical compound", "druggable (pocket or family)", "no tractability data"]


def gql(query, variables=None, tries=4):
    for attempt in range(tries):
        r = requests.post(URL, json={"query": query, "variables": variables or {}}, timeout=120)
        if r.ok and "errors" not in r.json():
            return r.json()["data"]
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Open Targets query failed: {r.status_code} {r.text[:300]}")


def map_symbols(symbols):
    """Gene symbol to Ensembl id, through the Open Targets search, 25 symbols per request."""
    out = {}
    symbols = sorted(set(symbols))
    for i in range(0, len(symbols), 25):
        chunk = symbols[i:i + 25]
        parts = [f's{j}: search(queryString: "{s}", entityNames: ["target"], page: {{index: 0, size: 3}}) '
                 f'{{ hits {{ id name }} }}' for j, s in enumerate(chunk)]
        data = gql("query { " + " ".join(parts) + " }")
        for j, s in enumerate(chunk):
            hits = data[f"s{j}"]["hits"]
            exact = [h for h in hits if h["name"].upper() == s.upper()]
            if exact:
                out[s] = exact[0]["id"]
            elif hits:
                out[s] = None  # a different gene came back; do not guess
            else:
                out[s] = None
        print(f"mapped {min(i + 25, len(symbols))}/{len(symbols)} symbols")
    return out


def fetch_targets(ensembl_ids):
    """Tractability, drugs and essentiality for each target, 20 targets per request."""
    rows = {}
    ids = sorted(set(i for i in ensembl_ids if i))
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        parts = [f't{j}: target(ensemblId: "{e}") {{ id approvedSymbol isEssential '
                 f'tractability {{ modality label value }} '
                 f'drugAndClinicalCandidates {{ count rows {{ maxClinicalStage drug {{ name drugType }} '
                 f'diseases {{ disease {{ name }} }} }} }} }}' for j, e in enumerate(chunk)]
        data = gql("query { " + " ".join(parts) + " }")
        for j, e in enumerate(chunk):
            t = data[f"t{j}"]
            if t is None:
                continue
            tr = {(x["modality"], x["label"]): x["value"] for x in t["tractability"]}
            cands = t["drugAndClinicalCandidates"]["rows"]
            stages = [c["maxClinicalStage"] for c in cands]
            drugs = []
            for c in sorted(cands, key=lambda c: -stage_rank(c["maxClinicalStage"]))[:4]:
                dis = sorted({d["disease"]["name"] for d in c["diseases"] if d.get("disease")})[:2]
                drugs.append(f"{c['drug']['name']} ({c['maxClinicalStage']}" + (f"; {', '.join(dis)}" if dis else "") + ")")
            rows[e] = {
                "sm_approved_drug": bool(tr.get(("SM", "Approved Drug"), False)),
                "sm_clinical": bool(tr.get(("SM", "Advanced Clinical"), False) or tr.get(("SM", "Phase 1 Clinical"), False)),
                "sm_pocket_or_family": bool(tr.get(("SM", "High-Quality Pocket"), False) or tr.get(("SM", "Med-Quality Pocket"), False)
                                            or tr.get(("SM", "Druggable Family"), False)),
                "ab_approved_drug": bool(tr.get(("AB", "Approved Drug"), False)),
                "ab_clinical": bool(tr.get(("AB", "Advanced Clinical"), False) or tr.get(("AB", "Phase 1 Clinical"), False)),
                "n_drugs_or_candidates": int(t["drugAndClinicalCandidates"]["count"]),
                "highest_clinical_stage": max(stages, key=stage_rank) if stages else "",
                "example_drugs": "; ".join(drugs),
                "common_essential_gene": bool(t["isEssential"]) if t["isEssential"] is not None else False,
            }
        print(f"fetched {min(i + 20, len(ids))}/{len(ids)} targets")
    return rows


def stage_rank(s):
    s = (s or "").lower()
    if "approved" in s:
        return 5
    for k, v in (("phase iv", 4), ("phase iii", 3), ("phase ii", 2), ("phase i", 1)):
        if k in s:
            return v
    return 0


def fetch_mb_associations():
    """Open Targets association score of every target with medulloblastoma."""
    scores = {}
    page = 0
    while True:
        q = """query($efo: String!, $page: Int!) { disease(efoId: $efo) {
                 associatedTargets(page: {index: $page, size: 500}) { count rows { score target { id } } } } }"""
        d = gql(q, {"efo": MB_EFO, "page": page})["disease"]["associatedTargets"]
        for r in d["rows"]:
            scores[r["target"]["id"]] = r["score"]
        page += 1
        if page * 500 >= d["count"]:
            break
    print(f"medulloblastoma associations: {len(scores)} targets")
    return scores


def tier(row):
    if row["sm_approved_drug"] or row["ab_approved_drug"]:
        return TIERS[0]
    if row["sm_clinical"] or row["ab_clinical"] or row["n_drugs_or_candidates"] > 0:
        return TIERS[1]
    if row["sm_pocket_or_family"]:
        return TIERS[2]
    return TIERS[3]


def annotate(df, sym2id, targets, mb):
    df = df.copy()
    df["ensembl_id"] = df["gene"].map(sym2id)
    df["mb_association_score"] = df["ensembl_id"].map(mb).fillna(0.0)
    cols = ["sm_approved_drug", "sm_clinical", "sm_pocket_or_family", "ab_approved_drug", "ab_clinical",
            "n_drugs_or_candidates", "highest_clinical_stage", "example_drugs", "common_essential_gene"]
    for c in cols:
        df[c] = df["ensembl_id"].map(lambda e: targets.get(e, {}).get(c, np.nan if c in ("n_drugs_or_candidates",) else ("" if c in ("highest_clinical_stage", "example_drugs") else False)))
    df["n_drugs_or_candidates"] = df["n_drugs_or_candidates"].fillna(0).astype(int)
    df["druggability_tier"] = df.apply(tier, axis=1)
    return df


def main():
    conc = pd.read_csv(TAB / "concordant_genes_per_group.csv")
    conc["direction"] = np.where(conc["meth_delta"] < 0, "promoter hypomethylated, expression up",
                                 "promoter hypermethylated, expression down")
    drivers = pd.DataFrame([(g, s) for g, l in DRIVERS.items() for s in l], columns=["group", "gene"])

    symbols = sorted(set(conc["gene"]) | set(drivers["gene"]))
    sym2id = map_symbols(symbols)
    missing = [s for s, e in sym2id.items() if e is None]
    print("symbols without an exact Open Targets match:", len(missing), missing[:20])
    targets = fetch_targets(sym2id.values())
    mb = fetch_mb_associations()

    ann = annotate(conc, sym2id, targets, mb)
    keep = ["group", "gene", "direction", "meth_delta", "meth_fdr", "expr_delta", "expr_fdr", "ensembl_id",
            "mb_association_score", "druggability_tier", "sm_approved_drug", "sm_clinical", "sm_pocket_or_family",
            "ab_approved_drug", "ab_clinical", "n_drugs_or_candidates", "highest_clinical_stage",
            "example_drugs", "common_essential_gene"]
    ann = ann[keep].sort_values(["group", "druggability_tier", "mb_association_score"], ascending=[True, True, False])
    ann.round(4).to_csv(TAB / "therapy_concordant_genes.csv", index=False)

    drv = annotate(drivers, sym2id, targets, mb)
    drv.round(4).to_csv(TAB / "therapy_driver_genes.csv", index=False)

    cand = ann[ann["druggability_tier"].isin(TIERS[:2])].copy()
    cand.round(4).to_csv(TAB / "therapy_candidates.csv", index=False)

    summary = {
        "concordant_genes": int(len(ann)),
        "genes_with_open_targets_entry": int(ann["ensembl_id"].notna().sum()),
        "per_group_tier_counts": {g: ann[ann["group"] == g]["druggability_tier"].value_counts().to_dict() for g in GROUPS},
        "per_group_switched_on_with_approved_or_clinical_drug": {
            g: sorted(cand[(cand["group"] == g) & cand["direction"].str.contains("expression up")]["gene"].tolist())
            for g in GROUPS},
        "per_group_switched_off_with_approved_or_clinical_drug": {
            g: sorted(cand[(cand["group"] == g) & cand["direction"].str.contains("expression down")]["gene"].tolist())
            for g in GROUPS},
        "drivers_with_approved_drug": sorted(drv[drv["sm_approved_drug"] | drv["ab_approved_drug"]]["gene"].unique().tolist()),
        "concordant_genes_with_mb_association_above_0.3": sorted(
            ann[ann["mb_association_score"] > 0.3][["group", "gene"]].apply(tuple, axis=1).tolist()),
    }
    json.dump(summary, open(TAB / "therapy_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # ---- figure: concordant genes per group by druggability tier and direction
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    palette = {TIERS[0]: "#1b7f3b", TIERS[1]: "#7fbf7b", TIERS[2]: "#c7c7c7", TIERS[3]: "#efefef"}
    for ax, d in zip(axes, ["promoter hypomethylated, expression up", "promoter hypermethylated, expression down"]):
        sub = ann[ann["direction"] == d]
        ct = sub.groupby(["group", "druggability_tier"]).size().unstack(fill_value=0).reindex(GROUPS).reindex(columns=TIERS, fill_value=0)
        left = np.zeros(len(GROUPS))
        for t in TIERS:
            ax.barh(GROUPS, ct[t].values, left=left, color=palette[t], edgecolor="white", label=t)
            left += ct[t].values
        ax.set_title(d, loc="left", fontsize=10)
        ax.set_xlabel("concordant genes")
        ax.invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine()
    fig.suptitle("Druggability of concordant genes (Open Targets tractability and clinical candidates)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "therapy_druggability.png", dpi=160)
    plt.close(fig)
    print(cand[["group", "gene", "direction", "druggability_tier", "highest_clinical_stage", "example_drugs"]].to_string(index=False))


if __name__ == "__main__":
    main()
