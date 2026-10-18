# Medulloblastoma subgroups from DNA methylation, with explanations

A learning project on public data. Seven pages, each a notebook executed end to end,
from the question through quality control, classification, explanation, the test against
gene expression, and the pathway and therapy angle. Code and tables:
[github.com/robinson-vidva/mb-methylation-xai](https://github.com/robinson-vidva/mb-methylation-xai).

- [1. The question and the data](01_question_and_data.html): Why medulloblastoma, why methylation, what the four questions are, and the data.
- [2. Quality control](02_quality_control.html): Probe filtering, beta distributions, failed probes in FFPE material, a sex check, and PCA.
- [3. Classification](03_classification.html): Three models, two tasks, cross-validation, the held-out split, calibration, an external cohort, and sparse coverage.
- [4. Explainability](04_explainability.html): Integrated gradients: which CpGs carry each subgroup call and which genes they sit in.
- [5. Methylation versus expression](05_methylation_vs_expression.html): Does promoter methylation at those sites track expression in the same tumours? Concordant genes per subgroup.
- [6. Pathways and therapy](06_pathways_and_therapy.html): Pathway enrichment, then Open Targets: which concordant genes and which drivers have drugs.
- [7. Conclusions](07_conclusions.html): Answers to the four questions, limits, reproducibility across operating systems, and next steps.

Data: GSE85212, GSE85217 (Cavalli et al. 2017), GSE54880 (Hovestadt et al. 2013), GPL13534.
