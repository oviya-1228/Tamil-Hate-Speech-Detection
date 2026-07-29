# Guns and Roses@DravidianCodeMix 2026

**A Target-Anchored Contrastive Learning Approach for Hierarchical Hate Speech Detection in Tamil**

Working notes submission for the DravidianCodeMix 2026 shared task. This repository contains the code, notebooks, and artifacts used to produce the results reported in our paper.

**Team members:**
- Mohnish Kanna K S
- Oviya T S
- Adithiyaa T
- Sarveash Koushik R V

## Overview

Hate speech in Tamil social media is often expressed implicitly, without slurs or profanity, which makes it hard for lexical classifiers to catch. Our system:

1. Fine-tunes a `google/muril-base-cased` backbone with a **target-anchored supervised contrastive loss** — hateful examples are pulled together by *target group* (e.g. all anti-women hate, explicit or implicit) rather than by surface wording, so the model can't rely on slur presence alone.
2. Combines this with a **LightGBM model** trained on character/word TF-IDF features, to catch slang, transliteration, and dialectal spelling the transformer may miss.
3. Calibrates both models' outputs with **isotonic regression** and combines them via a **logistic regression stacker**.
4. Outputs a **three-tier triage decision** (auto-flag / auto-clear / human review) instead of a bare binary label, for use in real moderation workflows.

## Results

| Configuration | Accuracy | HATE F1 | NON-HATE F1 | Implicit Recall |
|---|---|---|---|---|
| MuRIL, Level-1 only (baseline)* | 0.83 | 0.84 | 0.81 | ~0.00 |
| Contrastive MuRIL, default threshold | 0.79 | 0.78 | 0.80 | 0.68 |
| + F1-optimal threshold | 0.75 | 0.79 | 0.69 | 0.86 |
| + Isotonic calibration | 0.78 | 0.80 | 0.76 | 0.78 |
| + Lexical ensemble (final) | **0.79** | **0.81** | **0.77** | **0.78** |

*Baseline accuracy is inflated by an explicit-hate-majority error profile; it essentially ignores implicit hate.

**Level-wise hierarchical results:**

| Level | Task | Metric | Score | Notes |
|---|---|---|---|---|
| 1 | HATE / NON-HATE | Weighted F1 | 0.79 | Primary task |
| 3 | EXPLICIT / IMPLICIT | Macro F1 | 0.70 | True-HATE rows only |
| 2 | Target group | Macro F1 | 0.27 | Weak — see paper §6 |
| 5 | Bias category | Macro F1 | 0.15 | Insufficient data — see paper §7 |

Level 4 (fine-grained subcategory) was not modeled separately due to extreme class sparsity.

**Fairness check:** on true NON-HATE test examples, the false-positive rate was 0.620 (n=50) for sentences containing target-group vocabulary vs. 0.488 (n=252) for those without — a gap worth flagging as a limitation (see paper §7).

## Dataset

4,246 Tamil social-media sentences (after cleaning), annotated on a five-level hierarchy: HATE/NON-HATE → target group → EXPLICIT/IMPLICIT → fine-grained subcategory → bias category. Split 70/15/15 (train/val/test), stratified on Level-1 and Level-3 labels.

> Dataset not redistributed here — obtain it via the official DravidianCodeMix 2026 shared task release.

## Repository structure

```
.
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_baseline_diagnostic.ipynb        # Level-1-only MuRIL baseline
│   ├── 03_contrastive_training.ipynb       # target-anchored contrastive objective
│   ├── 04_lexical_ensemble.ipynb           # TF-IDF + LightGBM
│   ├── 05_calibration_and_stacking.ipynb   # isotonic regression + logistic stacker
│   ├── 06_triage_and_error_analysis.ipynb
│   └── 07_fairness_check.ipynb             # false-positive rate by keyword group
├── src/
│   ├── data.py
│   ├── model.py           # MuRIL + contrastive loss + multi-task heads
│   ├── ensemble.py        # LightGBM + stacking
│   └── triage.py
├── artifacts/
│   └── final_artifacts_v2.pkl
├── paper/
│   └── working_notes_paper.docx
└── README.md
```

> Adjust the tree above to match your actual repo layout before pushing.

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Key dependencies: `transformers`, `torch`, `lightgbm`, `scikit-learn`, `pandas`.

## Reproducing results

1. Run `01_data_cleaning.ipynb` to produce the cleaned, split CSVs.
2. Run `02_baseline_diagnostic.ipynb` to reproduce the diagnostic baseline (§5.1 of the paper).
3. Run `03_contrastive_training.ipynb` to fine-tune MuRIL with the target-anchored contrastive loss.
4. Run `04_lexical_ensemble.ipynb` and `05_calibration_and_stacking.ipynb` to build and calibrate the final ensemble.
5. Run `06_triage_and_error_analysis.ipynb` and `07_fairness_check.ipynb` to reproduce Tables in §5.3, §6, and §7.

Trained artifacts are cached in `artifacts/final_artifacts_v2.pkl` for anyone who wants to skip retraining and just reproduce the reported numbers.

## AI Assistance Disclosure

In accordance with the shared task's AI assistance policy: a large language model (Claude, Anthropic) was used as a code-writing assistant for the model architecture, training pipeline, and evaluation scripts, and was consulted during research framing and experimental design, including diagnosing the implicit-hate detection failure and designing the target-anchored contrastive objective. All reported results were independently generated by running the resulting code on the dataset described in the paper. Analysis, interpretation, and writing are the authors' own.

## Citation

```
Mohnish Kanna K S, Oviya T S, Adithiyaa T, Sarveash Koushik R V. (2026).
Guns and Roses@DravidianCodeMix 2026: A Target-Anchored Contrastive Learning
Approach for Hierarchical Hate Speech Detection in Tamil. Working Notes of
the DravidianCodeMix 2026 Shared Task, FIRE 2026.
```

## License

Code: add a license of your choice (MIT/Apache-2.0 are common for shared-task repos).
Paper: © 2026 the authors, CC BY 4.0.
