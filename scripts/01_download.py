"""Download public GEO files for the Cavalli 2017 and Hovestadt 2013 cohorts.

Run: python scripts/01_download.py
All files are public. Nothing here needs a login.
"""
import hashlib
import sys
import urllib.request

from common import RAW

BASE = "https://ftp.ncbi.nlm.nih.gov/geo"
FILES = {
    # Cavalli 2017, 763 medulloblastomas, Illumina 450k beta values
    "GSE85212_beta.txt.gz": (
        f"{BASE}/series/GSE85nnn/GSE85212/suppl/"
        "GSE85212_Methylation_763samples_SubtypeStudy_TaylorLab_beta_values.txt.gz",
        "0ab6cc50f4a838e8923f7c038c542fdb"),
    "GSE85212_series_matrix.txt.gz": (
        f"{BASE}/series/GSE85nnn/GSE85212/matrix/GSE85212_series_matrix.txt.gz", None),
    # Cavalli 2017, same 763 tumours, Affymetrix Human Gene 1.1 ST expression
    "GSE85217_expr.txt.gz": (
        f"{BASE}/series/GSE85nnn/GSE85217/suppl/"
        "GSE85217_M_exp_763_MB_SubtypeStudy_TaylorLab.txt.gz",
        "a2538879a0959bdef5d6af5b0f30dbab"),
    "GSE85217_series_matrix.txt.gz": (
        f"{BASE}/series/GSE85nnn/GSE85217/matrix/GSE85217_series_matrix.txt.gz", None),
    # Hovestadt 2013, 276 medulloblastomas plus 8 normal cerebellum, external test cohort
    "GSE54880_beta.txt.gz": (
        f"{BASE}/series/GSE54nnn/GSE54880/suppl/GSE54880_MB-450k_GEOupload_Beta.txt.gz",
        "f567f06f172d364af52423c73264b144"),
    "GSE54880_series_matrix.txt.gz": (
        f"{BASE}/series/GSE54nnn/GSE54880/matrix/GSE54880_series_matrix.txt.gz", None),
    # Illumina 450k manifest (probe to gene annotation)
    "GPL13534_manifest.csv.gz": (
        f"{BASE}/platforms/GPL13nnn/GPL13534/suppl/"
        "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz",
        "1c38681926364d725e394cb99ddda2cb"),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ok = True
    for name, (url, expected) in FILES.items():
        dest = RAW / name
        if not dest.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, dest)
        if expected is not None:
            got = md5(dest)
            status = "ok" if got == expected else "MISMATCH"
            ok &= got == expected
            print(f"{name}: md5 {got} {status}")
        else:
            print(f"{name}: present")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
