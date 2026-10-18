"""Shared paths, constants and small helpers."""
from pathlib import Path
import gzip
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
for _d in (RAW, PROC, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FEATURES = 10000

GROUPS = ["WNT", "SHH", "Group3", "Group4"]
SUBTYPES = [
    "WNT_alpha", "WNT_beta",
    "SHH_alpha", "SHH_beta", "SHH_gamma", "SHH_delta",
    "Group3_alpha", "Group3_beta", "Group3_gamma",
    "Group4_alpha", "Group4_beta", "Group4_gamma",
]
GROUP_COLORS = {"WNT": "#2a78d6", "SHH": "#eb6834", "Group3": "#1baf7a", "Group4": "#eda100"}
SUBTYPE_MARKERS = {"alpha": "o", "beta": "s", "gamma": "^", "delta": "D"}

PROMOTER_REGIONS = {"TSS200", "TSS1500", "5'UTR", "1stExon"}
BODY_REGIONS = {"Body", "3'UTR"}


def read_series_matrix(path):
    """Parse the sample annotation block of a GEO series matrix file."""
    rows = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, *vals = line.rstrip("\n").split("\t")
            vals = [v.strip('"') for v in vals]
            rows.setdefault(key, []).append(vals)
    out = pd.DataFrame({"geo_accession": rows["!Sample_geo_accession"][0],
                        "title": rows["!Sample_title"][0]})
    for vals in rows.get("!Sample_characteristics_ch1", []):
        keys = {v.split(": ", 1)[0] for v in vals if ": " in v}
        if len(keys) != 1:
            continue
        k = re.sub(r"[^a-z0-9]+", "_", keys.pop().lower()).strip("_")
        out[k] = [v.split(": ", 1)[1] if ": " in v else np.nan for v in vals]
    return out


def load_manifest():
    """Illumina 450k manifest from GEO (GPL13534), probe rows only."""
    cols = ["Name", "CHR", "MAPINFO", "Probe_SNPs", "UCSC_RefGene_Name",
            "UCSC_RefGene_Group", "Relation_to_UCSC_CpG_Island"]
    man = pd.read_csv(RAW / "GPL13534_manifest.csv.gz", skiprows=7, usecols=cols,
                      dtype=str, low_memory=False)
    man = man[man["Name"].str.startswith(("cg", "ch"), na=False)].set_index("Name")
    return man


def probe_gene_pairs(man, regions):
    """Long table of (probe, gene) pairs restricted to the given RefGene regions."""
    sub = man.dropna(subset=["UCSC_RefGene_Name"])
    genes = sub["UCSC_RefGene_Name"].str.split(";")
    groups = sub["UCSC_RefGene_Group"].str.split(";")
    recs = []
    for probe, gs, rs in zip(sub.index, genes, groups):
        seen = set()
        for g, r in zip(gs, rs):
            if r in regions and g not in seen:
                seen.add(g)
                recs.append((probe, g))
    return pd.DataFrame(recs, columns=["probe", "gene"])
