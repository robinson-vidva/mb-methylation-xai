"""Generate the seven report notebooks in notebooks/ and the page index docs/index.md.

The notebooks do not recompute anything. They read results/tables and results/figures written by
the numbered scripts. run_all.sh executes them and exports each one to docs/<name>.html.
"""
import nbformat as nbf

from common import ROOT

PAGES_URL = "https://robinson-vidva.github.io/mb-methylation-xai/"
PAPER = "Sharif Rahmani et al. 2023, MBMethPred, Frontiers in Genetics, https://doi.org/10.3389/fgene.2023.1233657"

PAGES = [
    ("01_question_and_data", "1. The question and the data"),
    ("02_quality_control", "2. Quality control"),
    ("03_classification", "3. Classification"),
    ("04_explainability", "4. Explainability"),
    ("05_methylation_vs_expression", "5. Methylation versus expression"),
    ("06_pathways_and_therapy", "6. Pathways and therapy"),
    ("07_gene_clusters_and_pathway_flow", "7. Gene clusters and pathway flow"),
    ("08_genes_roles_and_literature", "8. Genes: names, cancer roles and literature"),
    ("09_conclusions", "9. Conclusions"),
]

SETUP = ("import json\nimport pandas as pd\nfrom IPython.display import Image, display\n"
         "from pathlib import Path\nTAB = Path('../results/tables')\nFIG = Path('../results/figures')\n"
         "pd.set_option('display.width', 160)\npd.set_option('display.max_columns', 30)\n"
         "pd.set_option('display.max_colwidth', 90)")


def nav(i):
    prev_ = f"[previous: {PAGES[i-1][1]}]({PAGES_URL}{PAGES[i-1][0]}.html)" if i > 0 else ""
    next_ = f"[next: {PAGES[i+1][1]}]({PAGES_URL}{PAGES[i+1][0]}.html)" if i < len(PAGES) - 1 else ""
    parts = [p for p in (prev_, f"[index]({PAGES_URL})", next_) if p]
    return f"*Page {i+1} of {len(PAGES)}.* " + " | ".join(parts)


def build(i, cells):
    stem, title = PAGES[i]
    nb = nbf.v4.new_notebook()
    out = [nbf.v4.new_markdown_cell(f"# {title}\n\n{nav(i)}")]
    for kind, text in cells:
        out.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    out.append(nbf.v4.new_markdown_cell(nav(i)))
    nb["cells"] = out
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = ROOT / "notebooks" / f"{stem}.ipynb"
    nbf.write(nb, path)
    print("wrote", path)


# ----------------------------------------------------------------------------------------------
P1 = [
    ("md", """## Background

Medulloblastoma is the most common malignant brain tumour of childhood. It arises in the cerebellum.
Since 2016 the WHO classification splits it into four molecular subgroups: WNT, SHH, Group 3 and Group 4.
The subgroups differ in cell of origin, driver pathway, outcome and treatment. WNT tumours have an
excellent outcome and are candidates for therapy de-escalation. SHH tumours are driven by the
Hedgehog pathway, for which SMO inhibitors exist. Group 3 tumours often carry MYC amplification and have
the worst outcome. Group 4 is the largest subgroup and the least understood. Cavalli et al. 2017 split
the four subgroups further into twelve subtypes with distinct outcomes.

**DNA methylation.** A methyl group can be attached to a cytosine that is followed by a guanine, a CpG
site. Methylation at a gene promoter usually switches the gene off. Methylation patterns are set during
development, are stable, and survive formalin fixation. They encode the cell of origin of a tumour.
This is why methylation arrays became the clinical standard for classifying brain tumours
(Capper et al. 2018). The Illumina 450k array measures about 485,000 CpG sites. Each site gets a
**beta value**, the fraction of DNA copies that are methylated, from 0 (unmethylated) to 1 (methylated).

**Gene expression** is the amount of messenger RNA a gene produces. Here it was measured on an
Affymetrix array in the same 763 tumours. Having both measurements in the same tumours makes it possible
to test whether a methylation change actually changes a gene's output.

## Questions

1. Can a classifier trained on public beta values reproduce the published subgroup and subtype calls,
   and does it still work on tumours from another laboratory and on archival (FFPE) material?
2. Which CpG sites carry the decision, and which genes do they sit in?
3. Does promoter methylation at those sites track gene expression in the same tumours?
4. Which pathways differ between subgroups at the level of methylation and expression, and do any of
   the affected genes point to existing drugs?

Questions 1 and 2 are about diagnosis. Questions 3 and 4 are about biology and therapy. The link
between them is that a classifier is only useful beyond diagnosis if the sites it uses mean something.

## Data

All data are public and are fetched by `scripts/01_download.py`, with md5 checksums.

| Source | Content | Role here |
|---|---|---|
| GSE85212, Cavalli et al. 2017 | 763 primary medulloblastomas, 450k beta values | training and held-out test |
| GSE85217, Cavalli et al. 2017 | the same 763 tumours, Affymetrix expression | methylation versus expression |
| GSE54880, Hovestadt et al. 2013 | 276 medulloblastomas, 450k, 169 FFPE and 107 frozen | external test cohort |
| GPL13534 | Illumina 450k manifest | CpG to gene annotation |

Labels are the subgroup and subtype calls deposited with the data by the original authors.
They were themselves derived from methylation and expression clustering, so the classification part
of this project reproduces a known structure rather than discovering one.

The cohort below is split into 610 training and 153 test tumours, stratified by subtype, with a fixed
seed. The test tumours are not touched until the final evaluation."""),
    ("code", SETUP),
    ("code", "lab = pd.read_csv('../data/processed/labels.csv', index_col=0)\n"
             "pd.crosstab(lab['subtype'], lab['split'], margins=True)"),
    ("md", f"""The subtypes are unbalanced. Group 4 subtypes hold around a hundred tumours each. WNT beta holds 21
in total and 4 in the test split. This matters for every subtype result that follows.

## Related work

The idea was worked out from the data and the four questions above. For completeness: a published
R package, MBMethPred ({PAPER}), classifies the four subgroups from the same GSE85212 cohort and
explains its models with Shapley values. It is cited here as prior work, not as a template. The
subtype task, the gene-by-gene test against expression, calibration, sparse coverage and the drug
lookup are not part of it."""),
]

# ----------------------------------------------------------------------------------------------
P2 = [
    ("md", """Quality control answers one question before any modelling: is the data what it claims to be?
Four checks are run. Each is standard for methylation arrays and each is explained once.

## 1. Which CpG sites are kept, and why

Not every probe on the array is usable. Probes on the X and Y chromosomes track sex, not tumour biology.
Probes that overlap a common SNP measure the genotype, not methylation. The 450k array also carries
about 3,000 "ch" probes that measure non-CpG methylation. Probes with a missing value in any tumour
would need imputation. The table counts what survives each step."""),
    ("code", SETUP),
    ("code", "pd.read_csv(TAB / 'qc_probe_funnel.csv')"),
    ("md", """The deposited matrix already lacked the X and Y chromosomes and most of the array. The authors
had filtered it before upload. Removing SNP probes and ch probes leaves 308,020 CpGs, all complete,
all present in the external cohort as well. Modelling uses the 10,000 most variable of these,
chosen on the training split only so that the test split cannot leak into feature selection.

## 2. Does every tumour look like a methylation profile?

A healthy array gives a bimodal beta distribution: most sites are either unmethylated (near 0) or
methylated (near 1). A flat or shifted distribution signals a failed hybridisation or degraded DNA.
The figure overlays 15 random tumours per subgroup. The table summarises every tumour."""),
    ("code", "display(Image(FIG / 'qc_beta_density.png'))\n"
             "stats = pd.read_csv(TAB / 'qc_sample_stats.csv', index_col=0)\n"
             "stats.groupby('group')[['mean_beta', 'median_beta', 'sd_beta', 'fraction_below_0.2', 'fraction_above_0.8']].agg(['mean', 'min', 'max']).round(3)"),
    ("code", "display(Image(FIG / 'qc_sample_medians.png'))"),
    ("md", """All 763 tumours are bimodal and no tumour stands apart. Group 3 tumours have a slightly lower
mean beta than the other groups, which fits the known global hypomethylation of Group 3.

## 3. The external cohort: failed probes and FFPE

Hovestadt et al. 2013 deposited detection p-values alongside the beta values. A detection p-value
tests whether a probe's signal is above background. Probes with p above 0.01 are treated as missing
and were imputed with the training median before classification. Formalin-fixed material (FFPE)
fragments DNA and raises the failure rate, so the check is split by material."""),
    ("code", "det = pd.read_csv(TAB / 'qc_external_detection.csv', index_col=0)\n"
             "display(Image(FIG / 'qc_external_detection.png'))\n"
             "det.groupby('material')['fraction_failed_of_modelled_cpgs'].describe().round(4)"),
    ("md", """Frozen samples lose almost nothing. FFPE samples lose a median of 0.2 percent of the modelled CpGs,
and the worst sample loses 4.4 percent, below the 5 percent line at which a sample would be dropped.
Imputing these few sites with the training median is a small intervention. Section 3 of the
classification page shows that FFPE samples classify as well as frozen ones.

## 4. Sex check

Females carry two X chromosomes and one is inactivated by methylation. Mean methylation over X
chromosome CpGs is therefore higher in females. This is a cheap check that samples and metadata were
not swapped. The Cavalli deposit has no X probes and no sex annotation, so the check runs on the
external cohort, which has both."""),
    ("code", "display(Image(FIG / 'qc_external_sex_check.png'))\n"
             "pd.read_csv(TAB / 'qc_external_sex_check.csv', index_col=0).groupby('sex')['mean_beta_chrX'].describe().round(3)"),
    ("md", """Reported sex and X methylation agree for the large majority. A few FFPE samples sit between the two
clusters, which is expected with degraded DNA. No sample is on the wrong side by a clear margin.

## 5. Does the main structure of the data match the labels?

Principal component analysis (PCA) finds the directions of largest variation in the 10,000 modelled
CpGs. If the subgroups are real and the labels are right, they should separate along the first
components without any model. The right panel projects the external cohort onto the same axes to see
whether a different laboratory, and FFPE material, lands in the same place."""),
    ("code", "display(Image(FIG / 'qc_pca.png'))\npd.read_csv(TAB / 'qc_pca_variance.csv').T"),
    ("md", """The first two components carry almost half of the variance and separate the four subgroups
without supervision. WNT and SHH are tight. Group 3 and Group 4 touch, which is the known difficulty.
The external tumours, frozen and FFPE alike, fall on top of their Cavalli subgroups. There is no
visible laboratory or material batch effect on these axes, so a classifier trained on Cavalli has a
fair chance on Hovestadt."""),
]

# ----------------------------------------------------------------------------------------------
P3 = [
    ("md", """Three classifiers were trained on the 10,000 selected CpGs of the 610 training tumours: logistic
regression, a random forest, and a small neural network (a multilayer perceptron, MLP, two hidden
layers, dropout). Each was trained for two tasks: the four subgroups and the twelve subtypes.

**Balanced accuracy** is the mean of the per-class recalls. It weights every class equally, so a
rare subtype counts as much as a common one. Plain accuracy would be dominated by Group 4.
**Macro F1** is the mean of the per-class F1 scores. Both are reported.

## 1. Cross-validation and the held-out test split

Five-fold cross-validation on the training split gives an estimate of performance with a spread.
The held-out test split, 153 tumours never used for feature selection or training, gives one final
number per model."""),
    ("code", SETUP),
    ("code", "pd.read_csv(TAB / 'cv_summary.csv', header=[0, 1], index_col=[0, 1])"),
    ("code", "pd.read_csv(TAB / 'test_metrics.csv')"),
    ("md", """The four subgroups are close to solved by any model. This matches the literature.
The twelve subtypes are harder and the three models are within a few points of each other.
The confusion matrices show where the errors sit. Rows are the published label, columns the call."""),
    ("code", "display(Image(FIG / 'confusion_group_mlp.png'))\ndisplay(Image(FIG / 'confusion_subtype_mlp.png'))"),
    ("code", "rep = pd.read_csv(TAB / 'test_per_class.csv')\n"
             "rep[rep['model'] == 'mlp'].set_index(['task', 'class'])[['precision', 'recall', 'f1-score', 'support']]"),
    ("md", """Almost all subtype errors are inside SHH and WNT: SHH gamma against SHH alpha and beta, and WNT beta,
which has four test tumours. Group 3 and Group 4 subtypes are called well despite the overlap of the
two groups in the PCA.

## 2. Calibration

A classifier outputs a probability for each class. It is **calibrated** when a tumour called with
90 percent confidence is right 90 percent of the time. The expected calibration error (ECE) is the
average gap between confidence and observed accuracy across confidence bins. This matters in a
clinic: an overconfident model gives no warning on the cases it gets wrong."""),
    ("code", "display(Image(FIG / 'calibration_mlp.png'))\njson.load(open(TAB / 'train_summary.json'))"),
    ("md", """For the four subgroups the MLP is well calibrated: nearly all calls carry a confidence above 0.9 and
are nearly always right. For the twelve subtypes most calls still carry a confidence above 0.9 while
clearly fewer are right. The subtype model is overconfident. Before anyone trusted its scores it would
need temperature scaling or a similar correction.

## 3. External cohort, Hovestadt 2013

The models were trained on Cavalli only and applied to 276 tumours from a different laboratory,
with the four-group labels. The cohort is split by material."""),
    ("code", "pd.read_csv(TAB / 'external_validation.csv')"),
    ("code", "display(Image(FIG / 'confusion_group_mlp_external.png'))"),
    ("md", """Performance on the external cohort matches the internal test split. FFPE material classifies as well
as frozen material. This is the property that makes methylation usable on archival pathology blocks,
and it is the reason methylation, not expression, is the clinical reference for brain tumour classification.

## 4. Sparse coverage

Shallow nanopore sequencing is a cheaper and faster way to read methylation, with a result in hours
rather than days. It covers a random, sparse subset of CpGs. The simulation removes CpGs at random at
test time, 10 repeats per level, and measures four-group balanced accuracy. Missing CpGs are filled
with the training mean for the standard models. A third model, the same MLP, was trained with random
input masking, so that it learned to decide from whatever subset of CpGs is present."""),
    ("code", "display(Image(FIG / 'sparse_coverage.png'))"),
    ("md", """A model trained on complete arrays degrades at half coverage and collapses to chance once fewer than
a fifth of the CpGs are seen. The masked-input model keeps high balanced accuracy with a few hundred
of the 10,000 CpGs. The idea is simple and the effect is large. Two caveats: real nanopore coverage is
not uniform, and nanopore calls come from sequencing reads, not from array beta values.

## 5. Representation learning

A 16-dimensional autoencoder was trained on the training split. It compresses 10,000 beta values into
16 numbers and back. A linear classifier on the 16 numbers is compared with one on the raw CpGs, and
the 16 dimensions are drawn with UMAP."""),
    ("code", "display(Image(FIG / 'umap_latent.png'))\npd.read_csv(TAB / 'latent_vs_raw.csv')"),
    ("md", """The latent space separates the four subgroups cleanly and shows the subtypes as sub-clusters.
A linear classifier on 16 dimensions comes close to the raw CpGs for the subgroups and loses more for
the subtypes. The compression keeps the subgroup structure and drops some of the finer subtype signal."""),
]

# ----------------------------------------------------------------------------------------------
P4 = [
    ("md", """A classifier that is right is not yet useful for biology. The question here is which CpG sites
carry each subgroup call, and whether those sites sit in genes that make sense.

**Integrated gradients** attribute a prediction to the input features. The method moves the input from
a baseline (here the mean training profile) to the actual tumour in small steps and sums the gradient
of the predicted class along the path. The result is one number per CpG: positive when the tumour's
methylation at that site pushes the call towards the class, negative when it pushes away.
Attributions were computed on the four-group MLP and averaged over the correctly classified training
tumours of each subgroup.

Each CpG is annotated from the Illumina manifest with its gene, its position relative to the gene
(promoter regions TSS200, TSS1500, 5'UTR and first exon; or gene body), and whether the subgroup is
hyper- or hypomethylated at that site relative to the other subgroups."""),
    ("code", SETUP),
    ("code", "display(Image(FIG / 'top_cpgs_per_group.png'))"),
    ("code", "top = pd.read_csv(TAB / 'top_cpgs_per_group.csv')\n"
             "top.groupby('group').head(10)[['group', 'cpg', 'gene', 'region', 'cpg_island', 'direction', 'mean_attribution', 'delta_beta_vs_rest']]"),
    ("md", """The full list, 50 CpGs per subgroup, is in `results/tables/top_cpgs_per_group.csv`.
Two things stand out. First, most top CpGs are hypomethylated in their subgroup. Loss of promoter
methylation switches a gene on, so the model mostly recognises subgroups by programs that are turned
on, not off. Second, many top CpGs sit in genes of neuronal and developmental identity rather than in
the known driver genes. This is expected. Drivers such as CTNNB1, PTCH1 or MYC are defined by mutation
or amplification. Methylation records lineage, and lineage is what separates the subgroups.

Whether these sites actually control expression is the question of the next page."""),
]

# ----------------------------------------------------------------------------------------------
P5 = [
    ("md", """This page is what the matched expression data makes possible. A CpG can be highly discriminative
without doing anything. If promoter methylation at the model's top sites changes expression of the
gene, the explanation has biological content.

**Method.** For every gene, promoter methylation is the mean beta over its promoter probes (TSS200,
TSS1500, 5'UTR, first exon). The Spearman correlation (rho) between promoter methylation and expression
across all 763 tumours is computed for every gene with both measurements. A negative rho means that
more methylation goes with less expression, the expected direction for a promoter. The distribution
over all genes is the background. The genes under the model's top promoter CpGs are compared with it."""),
    ("code", SETUP),
    ("code", "display(Image(FIG / 'methylation_vs_expression_rho.png'))\njson.load(open(TAB / 'integration_summary.json'))"),
    ("code", "top = pd.read_csv(TAB / 'top_cpgs_with_expression_correlation.csv')\n"
             "top[top['promoter_cpg']].drop_duplicates('gene').sort_values('spearman_rho')[['group', 'gene', 'cpg', 'region', 'direction', 'spearman_rho']].head(15)"),
    ("md", """Across all genes the correlation is centred near zero: for most genes, promoter methylation on this
array does not track expression. For the genes under the model's top promoter CpGs it is clearly
negative, with most below -0.3. The sites the classifier relies on are, disproportionately, sites
where methylation does control expression. Examples with strong anti-correlation include VRK2 (WNT),
RBM24 and POU2F1 (Group 4) and AIPL1 (Group 3).

## Differential analysis per subgroup

The next step asks the same question subgroup by subgroup, for all genes rather than only the model's.
For each subgroup against the rest, a Mann-Whitney test is run on promoter methylation and on
expression for every gene. P-values are corrected for multiple testing with the Benjamini-Hochberg
false discovery rate (**FDR**); a gene with FDR below 0.05 is called significant. A gene is called
**concordant** when both tests are significant, the change in beta is above 0.10, the change in log2
expression is above 1 (a two-fold change), and the two changes point in opposite directions: promoter
hypomethylated with expression up, or promoter hypermethylated with expression down."""),
    ("code", "display(Image(FIG / 'differential_meth_vs_expr.png'))\n"
             "conc = pd.read_csv(TAB / 'concordant_genes_per_group.csv')\n"
             "conc['direction'] = conc['meth_delta'].lt(0).map({True: 'hypomethylated, expression up', False: 'hypermethylated, expression down'})\n"
             "pd.crosstab(conc['group'], conc['direction'], margins=True)"),
    ("md", """Each subgroup has tens to a few hundred concordant genes. Driver genes are circled in the figure.
GLI2 in SHH and OTX2 in Group 3 and Group 4 are concordant: their promoter is hypomethylated and they
are strongly expressed in their subgroup. Most other drivers are strongly over-expressed without a
promoter methylation change, which again says that drivers are switched by mutation or copy number,
not by methylation. The concordant genes are the input to the pathway and therapy page."""),
]

# ----------------------------------------------------------------------------------------------
P6 = [
    ("md", """## 1. Pathways

A single gene rarely explains a tumour. Pathway enrichment asks whether the concordant genes of a
subgroup cluster in a known biological process more often than chance. g:Profiler was queried with the
concordant genes per subgroup against a custom background of all genes that had both measurements,
with GO biological process, KEGG and Reactome as sources. Using the tested genes as background is
important: otherwise every list of brain tumour genes looks enriched for brain development."""),
    ("code", SETUP),
    ("code", "import os\nif os.path.exists(FIG / 'pathway_enrichment.png'):\n    display(Image(FIG / 'pathway_enrichment.png'))\n"
             "enr = pd.read_csv(TAB / 'pathway_enrichment_concordant_genes.csv')\n"
             "enr.groupby('query').head(6)[['query', 'source', 'name', 'p_value', 'intersection_size', 'term_size']]"),
    ("md", """Group 3 is dominated by a photoreceptor program: detection of light stimulus, visual perception,
phototransduction, with NRL, RCVRN, AIPL1, GNAT1 and IMPG2 among the genes. A photoreceptor identity
driven by the transcription factors NRL and CRX is a known and targetable feature of Group 3
medulloblastoma (Garancher et al. 2018). The analysis recovers it from methylation and expression
without being told. WNT returns the KEGG Wnt signalling pathway, the pathway that defines the
subgroup. SHH returns cell fate commitment. Group 4 returns no significant term, consistent with
Group 4 being the least understood subgroup.

## 2. From genes to drugs

Why methylation matters for therapy: a promoter that is hypomethylated in one subgroup switches a gene
on in that subgroup only. If the gene product can be blocked by a drug, the drug would act on those
tumours and spare the others. This is the logic behind subgroup-specific therapy. The reverse case,
a gene silenced by hypermethylation, can in principle be reactivated by demethylating agents such as
decitabine, which are approved in leukaemia.

The **Open Targets Platform** collects, for every human gene, whether an approved drug or a clinical
compound exists against its protein (**tractability**), the drugs and their highest clinical stage,
and an evidence score linking the gene to each disease. Every concordant gene and every curated
driver gene was queried through its public API. Genes were placed in four tiers: approved drug,
clinical compound, druggable (a binding pocket or a druggable protein family, but no compound yet),
or no tractability data.

This is hypothesis generation. A drug against a gene that is on in a subgroup is a lead to read about,
not a treatment. Most of the drugs below were developed for other diseases."""),
    ("code", "display(Image(FIG / 'therapy_druggability.png'))\n"
             "summ = json.load(open(TAB / 'therapy_summary.json'))\n"
             "pd.DataFrame(summ['per_group_tier_counts']).fillna(0).astype(int)"),
    ("code", "cand = pd.read_csv(TAB / 'therapy_candidates.csv')\n"
             "cand = cand[cand['direction'].str.contains('expression up')]\n"
             "cand.sort_values(['group', 'druggability_tier'])[['group', 'gene', 'druggability_tier', 'highest_clinical_stage', 'mb_association_score', 'example_drugs']].reset_index(drop=True)"),
    ("md", """**Reading the result.** Most concordant genes have no drug and no known pocket. The genes that do
have drugs are mostly ion channels, phosphodiesterases and extracellular matrix proteins, with drugs
developed for heart, nerve or fibrotic disease. Only one concordant gene, ZFHX3 in WNT, has an
Open Targets association with medulloblastoma above 0.3. A few leads are worth noting:

- **SYK** and **EPHA3** are switched on in WNT tumours. Both are kinases with clinical inhibitors
  (fostamatinib is approved; KB-004 targets EPHA3 in leukaemia and glioblastoma). WNT tumours already
  have an excellent outcome, so the clinical question there is de-escalation, not new drugs.
- **IGF1** is switched on in Group 3. IGF signalling has clinical antibodies (xentuzumab) and is a
  known growth signal in medulloblastoma models.
- **FZD7**, **FAP**, **LGALS1** and **COL1A2** are switched on in SHH. FZD7 is a Wnt receptor with a
  phase 1 antibody. FAP, LGALS1 and collagens mark the tumour stroma and the SHH subgroup is known to
  carry a desmoplastic stroma. These are markers of the microenvironment, not of the tumour cell.
- **MAPK10** (JNK3), switched on in SHH, has clinical JNK inhibitors.

The curated driver genes tell the complementary story."""),
    ("code", "drv = pd.read_csv(TAB / 'therapy_driver_genes.csv')\n"
             "drv[['group', 'driver_gene' if 'driver_gene' in drv.columns else 'gene', 'druggability_tier', 'highest_clinical_stage', 'mb_association_score', 'example_drugs']].rename(columns={'gene': 'driver_gene'})"),
    ("md", """Two drivers have approved drugs. **SMO**, the Hedgehog pathway receptor, is the target of vismodegib
and sonidegib, the one subgroup-specific therapy that has reached clinical trials in SHH medulloblastoma.
**CDK6**, amplified in a subset of Group 4, is the target of palbociclib and related inhibitors.
Neither driver is concordant: their promoters do not change. The two layers of the analysis therefore
point at different things. Mutated and amplified drivers are where approved drugs act today.
Methylation-controlled genes mark the lineage program and the microenvironment of each subgroup.
They are candidates for biomarkers and for drug repurposing, and, for Group 3, they recover the
photoreceptor dependency that is already under preclinical study."""),
]

# ----------------------------------------------------------------------------------------------
P7N = [
    ("md", """Pages 5 and 6 treat genes one at a time. This page asks how the concordant genes behave together,
and where the changed genes sit inside the pathways that define the subgroups. Four parts: clustering,
a network, pathway flow, and pathway activity per tumour.

## 1. Gene clusters

The concordant genes with both measurements were clustered by their promoter methylation across all
763 tumours (hierarchical clustering, average linkage, correlation distance, six clusters). The
expression panel uses the same gene order, so a block that is red (methylated) on the left should be
blue (silent) on the right if methylation controls expression."""),
    ("code", SETUP),
    ("code", "display(Image(FIG / 'gene_clusters_heatmap.png'))\n"
             "cl = pd.read_csv(TAB / 'gene_clusters.csv', index_col=0)\n"
             "pd.crosstab(cl['cluster'], cl['hypomethylated_in'])[['WNT', 'SHH', 'Group3', 'Group4']]"),
    ("code", "enr = pd.read_csv(TAB / 'gene_cluster_enrichment.csv')\n"
             "enr.groupby('query').head(6)[['query', 'source', 'name', 'p_value', 'intersection_size', 'term_size']]"),
    ("md", """The genes fall into subgroup blocks. One large cluster is hypomethylated in SHH and WNT tumours and
methylated in Group 3 and Group 4. Another is hypomethylated in Group 4, and a third in Group 3 and WNT.
The expression panel is the mirror image: where a block is unmethylated, the genes are expressed.
Methylation of these genes is organised by lineage, not gene by gene. Only one cluster returns
enriched terms, the Group 3 block, and the terms are visual perception and phototransduction: the
photoreceptor program again, now as a co-methylated block rather than a list.

## 2. Co-methylation network

A network makes the same point with less structure imposed. Nodes are concordant genes. An edge
joins two genes whose promoter methylation correlates across the 763 tumours with Spearman rho of
at least 0.6; red dashed edges mark anti-correlation of -0.6 or stronger. Communities were found by
greedy modularity. Node colour is the subgroup in which the gene is most hypomethylated; node size is
the number of edges. The five most connected genes per community are labelled."""),
    ("code", "display(Image(FIG / 'comethylation_network.png'))\n"
             "net = pd.read_csv(TAB / 'network_communities.csv')\n"
             "pd.crosstab(net['community'], net['hypomethylated_in'])[['WNT', 'SHH', 'Group3', 'Group4']]"),
    ("code", "pd.read_csv(TAB / 'network_hubs.csv')"),
    ("md", """Three large communities appear: two dominated by SHH genes and one by Group 3 and Group 4 genes.
The anti-correlated edges run between the SHH community and the Group 3 and Group 4 community: the
same gene is unmethylated in one lineage and methylated in the other. Hubs such as BOC (a Hedgehog
co-receptor), SEMA6A and KCNIP1 on the SHH side, and RBM24, CER1 and SAG on the Group 3 and 4 side,
are the genes whose methylation best summarises each block. These are candidate marker CpGs.

## 3. Pathway flow

A pathway is a chain: ligand, receptor, transducer, transcription factor, target genes. Where in the
chain a subgroup changes tells how the pathway is switched on. Three pathways were examined: Hedgehog
(the SHH subgroup), Wnt (the WNT subgroup), and the photoreceptor program (Group 3). Membership comes
from KEGG through its public API. The table counts, per pathway and subgroup, members with increased
or decreased expression, promoter changes, and concordant genes."""),
    ("code", "ms = pd.read_csv(TAB / 'pathway_membership_summary.csv')\n"
             "ms.set_index(['pathway', 'members', 'with_both_measurements', 'group'])"),
    ("md", """The figure places a curated core of each pathway in signalling order. Left block: change in promoter
beta per subgroup; right block: change in log2 expression. Numbers appear where the change passes
FDR 0.05 and the threshold used on page 5; a box marks a concordant gene."""),
    ("code", "display(Image(FIG / 'pathway_flow.png'))"),
    ("code", "flow = pd.read_csv(TAB / 'pathway_flow.csv')\n"
             "flow[flow['concordant']][['pathway', 'stage', 'gene', 'group', 'delta_promoter_beta', 'delta_log2_expression']].reset_index(drop=True)"),
    ("md", """**Hedgehog in SHH tumours.** The output of the pathway is on: PTCH1, PTCH2, HHIP, GLI1, GLI2, MYCN,
CCND1 and SFRP1 are all over-expressed. Almost none of their promoters change; only BOC and GLI2 are
concordant. The ligand SHH, the transducers SMO, SUFU and KIF7 show nothing at either level. This is
the signature of a pathway switched on by mutation (PTCH1, SMO or SUFU loss, upstream of the targets):
the signal appears as target gene output, and methylation is silent.

**Wnt in WNT tumours.** The same pattern. AXIN2, LEF1, NKD1, DKK1, DKK2, WIF1, SP5, NOTUM and MYC are
strongly over-expressed, the classic beta-catenin target set, while the promoters of CTNNB1, APC and
the destruction complex do not move. CTNNB1 mutation drives the subgroup, and the targets respond at
the expression level. Only SP5 is concordant.

**Photoreceptor program in Group 3.** Here the chain changes at every level and in both measurements.
The transcription factors OTX2, NRL and NR2E3 are hypomethylated and expressed; the phototransduction
genes RHO, GNAT1, GNGT1, PDE6H, SAG and RCVRN follow; the structural genes AIPL1, IMPG2 and PRPH2
follow too. Seven of the 26 measured members are concordant in Group 3. The mirror image holds in SHH
tumours, where OTX2, SAG, AIPL1, IMPG2 and PRPH2 are hypermethylated and silent. This is a program
controlled by promoter methylation, not by a mutation.

## 4. Pathway activity per tumour

The same contrast, reduced to one number per tumour: the mean expression of each pathway's target
genes, and the mean promoter methylation of the same genes."""),
    ("code", "display(Image(FIG / 'pathway_activity.png'))\n"
             "act = pd.read_csv(TAB / 'pathway_activity_scores.csv', index_col=0)\n"
             "act.groupby('group')[[c for c in act.columns if '|' in c]].median().round(3).T"),
    ("md", """Hedgehog and Wnt target expression is specific to SHH and WNT tumours, and the promoter methylation
of those targets is flat across subgroups. The photoreceptor program is high in Group 3 and its
promoters are clearly less methylated there than in every other subgroup.

**What this means for therapy.** A pathway switched on by mutation is targeted at the mutated node.
That is why SMO inhibitors work in SHH tumours and why the methylation state of Hedgehog targets adds
nothing to that decision. A program switched on by demethylation is a different kind of target. Its
promoter methylation is a per-tumour readout that could serve as a biomarker, and the program itself,
here the NRL and CRX photoreceptor identity of Group 3, is a dependency that has been proposed as a
therapeutic entry point. Which Group 3 tumours carry the program, and how strongly, is visible in the
top right panel. This is as far as array data can go; it does not test any drug."""),
]

# ----------------------------------------------------------------------------------------------
P8 = [
    ("md", """The previous pages name genes by symbol. This page says what each gene is, whether it is a known
cancer gene, what the literature says about it in medulloblastoma, and which signalling pathways it
belongs to. It then reads every concordant change against that knowledge.

**Sources, all public.** Full names and the NCBI gene summary come from MyGene.info. Oncogene and tumour
suppressor status comes from two curated lists: the OncoKB cancer gene list (gene type: oncogene,
tumour suppressor, both, or neither) and the COSMIC Cancer Gene Census "role in cancer", read through
the Open Targets Platform together with the COSMIC cancer hallmarks. Pathway membership is Reactome,
also through Open Targets. Literature is a PubMed count of papers that mention the gene together with
medulloblastoma in title or abstract, and the three most relevant of those papers.

**The reading.** A promoter that loses methylation switches a gene on; a promoter that gains
methylation switches it off. Two combinations fit a causal story in cancer: an **oncogene switched
on** by hypomethylation, and a **tumour suppressor silenced** by hypermethylation. The other two, an
oncogene silenced or a tumour suppressor switched on, do not, and are more likely lineage markers
that happen to carry a cancer label from another tissue. Most concordant genes have no cancer role in
either list; for them the methylation change reports cell identity, not transformation."""),
    ("code", SETUP),
    ("code", "ann = pd.read_csv(TAB / 'gene_annotation.csv', index_col=0)\n"
             "summ = json.load(open(TAB / 'gene_annotation_summary.json'))\n"
             "print('genes annotated:', summ['genes_annotated'], '| with a cancer role:', summ['with_cancer_role'])\n"
             "pd.Series(summ['cancer_role_counts'], name='genes').to_frame()"),
    ("md", """## 1. The reading per subgroup"""),
    ("code", "display(Image(FIG / 'gene_roles_by_group.png'))\n"
             "reg = pd.read_csv(TAB / 'gene_regulation_concordant.csv')\n"
             "pd.crosstab(reg['group'], reg['reading']).reindex(['WNT', 'SHH', 'Group3', 'Group4'])"),
    ("md", """The genes that fit the causal story, per subgroup, with their names, the evidence source and the
number of medulloblastoma papers."""),
    ("code", "cols = ['group', 'gene', 'gene_name', 'direction', 'oncokb_gene_type', 'cgc_role_in_cancer', 'pubmed_medulloblastoma', 'reading']\n"
             "fits = reg[reg['reading'].isin(['oncogene switched on', 'tumour suppressor silenced'])]\n"
             "fits.sort_values(['group', 'pubmed_medulloblastoma'], ascending=[True, False])[cols].reset_index(drop=True)"),
    ("md", """And the genes that carry a cancer label but change in the other direction. These are the ones to be
careful with: a label from leukaemia or breast cancer does not transfer to a cerebellar tumour."""),
    ("code", "against = reg[reg['reading'].isin(['oncogene silenced', 'tumour suppressor switched on', 'dual role gene changed'])]\n"
             "against.sort_values(['group', 'pubmed_medulloblastoma'], ascending=[True, False])[cols].reset_index(drop=True)"),
    ("md", """## 2. Signalling map

Concordant genes that have a cancer role or sit in a Reactome signal transduction pathway, per
subgroup. Position is the expression change, colour the direction of the methylation change, dot size
the number of medulloblastoma papers. Genes without any paper are the unexplored part of each list."""),
    ("code", "display(Image(FIG / 'signalling_map.png'))"),
    ("md", """## 3. The most studied genes in each subgroup

Literature count is a crude measure of what is known. A concordant gene with many medulloblastoma
papers is a known marker and the methylation result confirms it. A concordant gene with none is either
new or unimportant; the table of all genes at the end lets the reader decide."""),
    ("code", "pd.DataFrame({g: [f\"{x[0]} ({int(x[1])})\" for x in summ['most_cited_in_medulloblastoma_per_group'][g]] for g in ['WNT', 'SHH', 'Group3', 'Group4']})"),
    ("code", "lit = pd.read_csv(TAB / 'gene_literature_top.csv')\n"
             "topg = reg.sort_values('pubmed_medulloblastoma', ascending=False).drop_duplicates('gene').head(12)['gene']\n"
             "lit[lit['gene'].isin(topg) & (lit['rank'] == 1)].merge(reg.drop_duplicates('gene')[['gene', 'group', 'direction']], on='gene')[['group', 'gene', 'direction', 'year', 'title']].reset_index(drop=True)"),
    ("md", """## 4. The driver genes, annotated

The curated driver genes of each subgroup with the same annotation, for comparison. They are the
mutated or amplified genes; their methylation does not change (page 5), but their roles and literature
anchor the subgroup biology."""),
    ("code", "drv = pd.read_csv(TAB / 'therapy_driver_genes.csv')[['group', 'gene']]\n"
             "drv.merge(ann[['gene_name', 'cancer_role', 'oncokb_gene_type', 'cgc_role_in_cancer', 'pubmed_medulloblastoma', 'reactome_top_level']], left_on='gene', right_index=True)"),
    ("md", """## 5. Reading the biology, subgroup by subgroup

**First, the overall count.** Of 464 annotated genes, 46 carry a curated cancer role. Of the 325
concordant genes, 262 have never been mentioned together with medulloblastoma on PubMed. Both numbers
say the same thing as page 7: the genes whose methylation defines a subgroup are mostly lineage genes,
not cancer genes. That is expected for a tumour classified by cell of origin, and it is the reason the
classifier works on FFPE blocks from another laboratory.

**WNT.** This is the subgroup where the reading has the most content, and the most instructive case
is the one that looks wrong. Six tumour suppressors are switched on: RNF43, ZNRF3, APCDD1, ZFHX3,
EPHA3 and SLFN11. RNF43 and ZNRF3 are E3 ubiquitin ligases that remove Frizzled receptors from the
cell surface; APCDD1 is a membrane Wnt inhibitor. All three are transcriptional targets of
beta-catenin, together with AXIN2, NKD1, NOTUM and DKK1 from page 7. They form the negative feedback
loop of the pathway: when Wnt signalling fires, the cell induces its own brakes. In WNT
medulloblastoma, CTNNB1 carries a mutation in the degradation motif, so beta-catenin is stable no
matter what happens at the receptor. The brakes are induced, hypomethylated and expressed, and they
have nothing to act on. "Tumour suppressor switched on" is here the signature of a pathway stuck on
downstream of the point where the suppressor acts. The one tumour suppressor silenced, SFRP1, a
secreted Wnt antagonist, is the classic promoter hypermethylation event of Wnt-driven cancers, and
has ten medulloblastoma papers. The oncogenes switched on, SYK and STAT4, are immune signalling
kinases and transcription factors with no medulloblastoma literature; they may reflect the immune
and stromal content of WNT tumours rather than the tumour cell, and are leads at most.

**SHH.** Almost nothing with a cancer label is concordant. GLI2 and BOC are the pathway-level hits
and carry the most literature (70 and 3 papers). APCDD1 silenced and NDRG1 switched on are the only
labelled genes; NDRG1 is a hypoxia and differentiation gene. The SHH concordant list is dominated by
genes of the desmoplastic stroma and of neuronal differentiation. The Hedgehog pathway itself is
activated by mutation, as page 7 shows, and the methylation layer adds the context of the lineage,
not the driver.

**Group 3.** IGF1 is the one oncogene switched on, with 15 medulloblastoma papers; insulin-like
growth factor signalling is an established growth signal in Group 3 models and a candidate target.
The rest of the labelled list is the photoreceptor program (RHO, GNAT1, GNGT1, GNGT2 in the
signalling map), which is lineage identity, and GLI2 and OTX2 silenced or switched on as the mirror
image of SHH and Group 4.

**Group 4.** No cancer-role gene fits the causal story. DDR2 and SCG5 are silenced, which reads as
lineage again. The most cited concordant genes, OTX2, EYA1, EOMES and RBM24, are developmental
transcription factors and RNA-binding proteins of the cerebellar lineage. Group 4 stays what the
literature says it is: the subgroup defined by its cell of origin rather than by a driver pathway,
with KDM6A, PRDM6 and SNCAIP as mutated or rearranged genes whose promoters do not change.

**Caveats on the counts.** PubMed counts use the gene symbol in title or abstract, so symbols that are
also common abbreviations are inflated: FAP (familial adenomatous polyposis, itself linked to
medulloblastoma), SAG (sagittal), SST, TF and RHO. Those counts should be read as upper bounds.
Twenty-nine symbols come from the 2011 Illumina manifest and are no longer current (C1orf61,
HIST1H3E, SEPP1 and others); they have no name in the table. Oncogene and tumour suppressor labels
come from pan-cancer curation and do not transfer automatically to a cerebellar tumour of childhood;
the WNT feedback genes are the proof."""),
    ("md", """## 6. Every annotated gene

The full table, one row per gene: name, which analyses it appears in, cancer role and sources,
hallmarks, Reactome pathways, PubMed counts, the Open Targets medulloblastoma association score and
the NCBI summary. Also in `results/tables/gene_annotation.csv`."""),
    ("code", "pd.set_option('display.max_rows', 500)\n"
             "ann.sort_values('pubmed_medulloblastoma', ascending=False)[['gene_name', 'in_sets', 'cancer_role', 'oncokb_gene_type', 'cgc_role_in_cancer', 'pubmed_medulloblastoma', 'pubmed_cancer', 'reactome_top_level']]"),
]

# ----------------------------------------------------------------------------------------------
P7 = [
    ("md", f"""## Answers to the four questions

1. **Classification.** Public beta values and a fixed pipeline reproduce the four subgroups at
   balanced accuracy 0.97 on the held-out split and 0.98 on an external cohort, with FFPE as good as
   frozen. The twelve subtypes reach 0.86 at best. The subtype model is overconfident. Trained with
   random input masking, the same network still classifies subgroups from a few hundred CpGs.
2. **Explanations.** The CpGs that drive the four-group calls are mostly hypomethylated in their
   subgroup and sit in genes of neuronal and developmental identity, not in the known drivers.
3. **Methylation and expression.** Across all genes, promoter methylation barely tracks expression.
   At the model's top promoter sites it does, with a clearly negative correlation for most genes.
   The classifier's explanations point at functional sites.
4. **Pathways and therapy.** Concordant genes recover the Group 3 photoreceptor program, Wnt
   signalling in WNT and cell fate commitment in SHH. Few concordant genes have drugs, and those that do
   are mostly channels and stromal genes. The approved subgroup-specific drugs act on mutated drivers
   (SMO, CDK6) whose promoters do not change. The pathway flow analysis makes the distinction explicit:
   Hedgehog and Wnt targets are over-expressed in SHH and WNT tumours with flat promoter methylation,
   the signature of activation by mutation, while the Group 3 photoreceptor program is hypomethylated
   and expressed along the whole chain, the signature of activation by demethylation. Methylation marks
   lineage programs and microenvironment; mutation marks the current drug targets.

## What this project does not show

- Labels came from clustering of the same data types, so the classification is a reproduction.
- Beta values were taken as deposited. The clinical reference classifier starts from raw IDAT files
  with its own normalisation.
- Variance-based feature selection and a 10,000-CpG cap are simple choices.
- Correlation between methylation and expression is not causation.
- The sparse-coverage simulation removes CpGs uniformly. Real nanopore coverage is not uniform.
- 153 test tumours, 4 of them WNT beta. Subtype numbers are noisy.
- Open Targets tractability says a drug exists against a protein, not that the drug helps in
  medulloblastoma or crosses the blood-brain barrier.

## Reproducibility

Seeds are fixed. The pipeline was run twice from the same code and GEO files, on Linux and on
Windows. Every result from scikit-learn, scipy and statsmodels was identical. Every result that
passes through PyTorch (the MLP, the autoencoder, the integrated gradients) differed, because the two
CPU builds sum floating point numbers in a different order and 80 training epochs amplify that.
The largest shift was the twelve-subtype MLP, 0.859 against 0.816 balanced accuracy. The README
lists every difference. A stronger version would train several seeds and report the spread."""),
    ("code", SETUP),
    ("code", "pd.read_csv(TAB / 'test_metrics.csv')"),
    ("md", """## What would come next

- Start from IDAT files with a standard normalisation, and compare with the deposited beta values.
- Train several seeds and report means with spread for every PyTorch result.
- Temperature scaling for the subtype model, then check calibration on the external cohort.
- Survival: the Cavalli subtypes differ in outcome, and the same cohort allows CpG-level markers to
  be tested against survival.
- Nanopore: replace the uniform simulation with real shallow-sequencing coverage patterns.
- Therapy: for the Group 3 photoreceptor program and the SHH stromal genes, read the preclinical
  literature and the DepMap dependency data, and test whether the concordant genes separate
  responders in any published drug screen on medulloblastoma lines."""),
]


def main():
    for i, cells in enumerate([P1, P2, P3, P4, P5, P6, P7N, P8, P7]):
        build(i, cells)
    lines = ["# Medulloblastoma subgroups from DNA methylation, with explanations", "",
             "A learning project on public data. Nine pages, each a notebook executed end to end,",
             "from the question through quality control, classification, explanation, the test against",
             "gene expression, and the pathway and therapy angle. Code and tables:",
             "[github.com/robinson-vidva/mb-methylation-xai](https://github.com/robinson-vidva/mb-methylation-xai).", ""]
    blurbs = [
        "Why medulloblastoma, why methylation, what the four questions are, and the data.",
        "Probe filtering, beta distributions, failed probes in FFPE material, a sex check, and PCA.",
        "Three models, two tasks, cross-validation, the held-out split, calibration, an external cohort, and sparse coverage.",
        "Integrated gradients: which CpGs carry each subgroup call and which genes they sit in.",
        "Does promoter methylation at those sites track expression in the same tumours? Concordant genes per subgroup.",
        "Pathway enrichment, then Open Targets: which concordant genes and which drivers have drugs.",
        "Gene clusters, a co-methylation network, pathway flow for Hedgehog, Wnt and the photoreceptor program, pathway activity per tumour.",
        "Every gene named: full name, oncogene or tumour suppressor status, medulloblastoma literature, Reactome pathways, and whether the methylation change fits the role.",
        "Answers to the four questions, limits, reproducibility across operating systems, and next steps.",
    ]
    for (stem, title), blurb in zip(PAGES, blurbs):
        lines.append(f"- [{title}]({stem}.html): {blurb}")
    lines += ["", "Data: GSE85212, GSE85217 (Cavalli et al. 2017), GSE54880 (Hovestadt et al. 2013), GPL13534.", ""]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("wrote", ROOT / "docs" / "index.md")


if __name__ == "__main__":
    main()
