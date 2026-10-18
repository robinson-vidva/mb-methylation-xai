"""Train and evaluate subgroup and subtype classifiers.

Experiments:
  1. 5-fold cross-validation on the training split, three models, two tasks
  2. held-out test split, per-class report and confusion matrices
  3. calibration of the MLP on the test split
  4. external validation on Hovestadt 2013 (GSE54880), overall and by FFPE versus frozen
  5. sparse-coverage simulation: accuracy when most CpGs are missing at test time
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold

from common import FIG, GROUPS, GROUP_COLORS, PROC, SEED, SUBTYPES, TAB
from models import predict_proba, train_mlp

sns.set_theme(style="white", font_scale=0.9)
TASKS = {"group": GROUPS, "subtype": SUBTYPES}


def fit_predict(name, Xtr, ytr, Xte, n_classes, **kw):
    if name == "logreg":
        m = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED).fit(Xtr, ytr)
        return m.predict_proba(Xte)
    if name == "rf":
        m = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        return m.predict_proba(Xte)
    if name == "mlp":
        m = train_mlp(Xtr, ytr, n_classes, **kw)
        return predict_proba(m, Xte)
    raise ValueError(name)


def metrics(y, p):
    return {"balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "accuracy": float(np.mean(y == p))}


def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        rows.append((lo, hi, m.sum(), acc, c))
    return ece, pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "n", "accuracy", "confidence"])


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(0.55 * len(classes) + 2.5, 0.5 * len(classes) + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette("#2a78d6", as_cmap=True),
                cbar=False, xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=1, linecolor="white")
    ax.set_xlabel("predicted")
    ax.set_ylabel("pathology label")
    ax.set_title(title, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    X = pd.read_parquet(PROC / "beta_top.parquet")
    lab = pd.read_csv(PROC / "labels.csv", index_col=0).loc[X.index]
    tr, te = lab["split"] == "train", lab["split"] == "test"
    Xtr, Xte = X.values[tr.values], X.values[te.values]
    summary = {}

    # ---- 1. cross-validation and 2. test split
    cv_rows, test_rows, report_rows, calibration = [], [], [], {}
    for task, classes in TASKS.items():
        enc = {c: i for i, c in enumerate(classes)}
        y = lab[task].map(enc).values
        ytr, yte = y[tr.values], y[te.values]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for model in ["logreg", "rf", "mlp"]:
            for fold, (a, b) in enumerate(skf.split(Xtr, ytr)):
                proba = fit_predict(model, Xtr[a], ytr[a], Xtr[b], len(classes))
                cv_rows.append({"task": task, "model": model, "fold": fold,
                                **metrics(ytr[b], proba.argmax(1))})
            proba = fit_predict(model, Xtr, ytr, Xte, len(classes))
            pred = proba.argmax(1)
            test_rows.append({"task": task, "model": model, **metrics(yte, pred)})
            cm = confusion_matrix(yte, pred, labels=range(len(classes)))
            plot_confusion(cm, classes, f"{task}, {model}, held-out test (n={len(yte)})",
                           FIG / f"confusion_{task}_{model}.png")
            rep = classification_report(yte, pred, labels=range(len(classes)),
                                        target_names=classes, output_dict=True, zero_division=0)
            for c in classes:
                report_rows.append({"task": task, "model": model, "class": c, **rep[c]})
            if model == "mlp":
                np.save(PROC / f"test_proba_{task}.npy", proba)
                conf, correct = proba.max(1), (pred == yte).astype(float)
                ece, bins = expected_calibration_error(conf, correct)
                summary[f"mlp_{task}_test_ece"] = float(ece)
                summary[f"mlp_{task}_test_fraction_confidence_above_0.9"] = float((conf > 0.9).mean())
                bins.to_csv(TAB / f"calibration_bins_{task}_mlp.csv", index=False)
                calibration[task] = (ece, bins)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (task, (ece, bins)) in zip(axes, calibration.items()):
        ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--")
        ax.scatter(bins["confidence"], bins["accuracy"], s=12 + 2.5 * bins["n"], color="#2a78d6",
                   edgecolor="white", linewidth=0.6, zorder=3)
        for _, r in bins.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["confidence"], r["accuracy"]),
                        textcoords="offset points", xytext=(-14, 10), fontsize=7)
        ax.set_xlabel("mean predicted confidence in bin")
        ax.set_ylabel("observed accuracy in bin")
        ax.set_title(f"MLP, {len(TASKS[task])} {task}s, ECE {ece:.3f}", loc="left")
        ax.set_xlim(0.2, 1.05)
        ax.set_ylim(0.2, 1.05)
    sns.despine()
    fig.suptitle("Calibration on the held-out test split (dot size = tumours in bin)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_mlp.png", dpi=160)
    plt.close(fig)

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(TAB / "cv_folds.csv", index=False)
    cv.groupby(["task", "model"])[["balanced_accuracy", "macro_f1", "accuracy"]].agg(
        ["mean", "std"]).round(3).to_csv(TAB / "cv_summary.csv")
    pd.DataFrame(test_rows).round(3).to_csv(TAB / "test_metrics.csv", index=False)
    pd.DataFrame(report_rows).round(3).to_csv(TAB / "test_per_class.csv", index=False)

    # ---- final group and subtype MLPs on the whole training split, saved for explanation
    for task, classes in TASKS.items():
        enc = {c: i for i, c in enumerate(classes)}
        ytr = lab.loc[tr, task].map(enc).values
        m = train_mlp(Xtr, ytr, len(classes))
        torch.save(m.state_dict(), PROC / f"mlp_{task}.pt")

    # ---- 4. external validation, Hovestadt 2013
    ext = pd.read_parquet(PROC / "external_beta_top.parquet")[X.columns]
    ext_lab = pd.read_csv(PROC / "external_labels.csv", index_col=0).loc[ext.index]
    keep = ext_lab["group"].isin(GROUPS)
    ext, ext_lab = ext[keep.values], ext_lab[keep]
    enc = {c: i for i, c in enumerate(GROUPS)}
    yext = ext_lab["group"].map(enc).values
    ytr_g = lab.loc[tr, "group"].map(enc).values
    ext_rows, ext_pred = [], {}
    for model in ["logreg", "rf", "mlp"]:
        proba = fit_predict(model, Xtr, ytr_g, ext.values, len(GROUPS))
        pred = proba.argmax(1)
        ext_pred[model] = pred
        ext_rows.append({"model": model, "subset": "all", "n": len(yext), **metrics(yext, pred)})
        for mat in ["frozen", "FFPE"]:
            m = (ext_lab["material"] == mat).values
            ext_rows.append({"model": model, "subset": mat, "n": int(m.sum()),
                             **metrics(yext[m], pred[m])})
    pd.DataFrame(ext_rows).round(3).to_csv(TAB / "external_validation.csv", index=False)
    plot_confusion(confusion_matrix(yext, ext_pred["mlp"], labels=range(4)), GROUPS,
                   f"4 groups, MLP, external cohort GSE54880 (n={len(yext)})",
                   FIG / "confusion_group_mlp_external.png")

    # ---- 5. sparse coverage: CpGs missing at test time
    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    rng = np.random.default_rng(SEED)
    yte_g = lab.loc[te, "group"].map(enc).values
    train_mean = Xtr.mean(axis=0)
    models = {
        "logistic regression": LogisticRegression(max_iter=3000, C=0.5, random_state=SEED).fit(Xtr, ytr_g),
        "MLP": train_mlp(Xtr, ytr_g, 4),
        "MLP, masked-input training": train_mlp(Xtr, ytr_g, 4, input_mask_max=0.98, epochs=120),
    }
    cov_rows = []
    for frac in fractions:
        for rep in range(10):
            keep_mask = rng.random(Xte.shape[1]) < frac
            Xm = np.where(keep_mask, Xte, train_mean)
            Xz = np.where(keep_mask, Xte, 0.0)
            for name, m in models.items():
                Xin = Xz if "masked" in name else Xm
                p = (predict_proba(m, Xin) if name.startswith("MLP") else m.predict_proba(Xin)).argmax(1)
                cov_rows.append({"model": name, "fraction_cpgs_observed": frac, "rep": rep,
                                 "n_cpgs": int(keep_mask.sum()),
                                 "balanced_accuracy": balanced_accuracy_score(yte_g, p)})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(TAB / "sparse_coverage.csv", index=False)
    agg = cov.groupby(["model", "fraction_cpgs_observed"])["balanced_accuracy"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"logistic regression": "#2a78d6", "MLP": "#eb6834", "MLP, masked-input training": "#1baf7a"}
    for name, g in agg.groupby("model"):
        ax.errorbar(g["fraction_cpgs_observed"] * 10000, g["mean"], yerr=g["std"], marker="o",
                    ms=5, lw=2, capsize=2, color=palette[name], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("CpGs observed at test time (of 10,000)")
    ax.set_ylabel("balanced accuracy, 4 groups, test split")
    ax.set_title("Robustness to sparse coverage", loc="left")
    ax.set_ylim(0.2, 1.02)
    ax.legend(frameon=False, loc="lower right")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG / "sparse_coverage.png", dpi=160)
    plt.close(fig)

    json.dump(summary, open(TAB / "train_summary.json", "w"), indent=2)
    print(pd.read_csv(TAB / "test_metrics.csv"))
    print(pd.read_csv(TAB / "external_validation.csv"))


if __name__ == "__main__":
    main()
