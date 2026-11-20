# Guns and Roses @ DravidianCodeMix 2026

## A Target-Anchored Contrastive Learning Approach for Hierarchical Hate Speech Detection in Tamil

This repository contains the implementation, Jupyter notebooks, experimental artifacts, and working notes for our submission to the **DravidianCodeMix 2026 Shared Task**, part of FIRE 2026.

Our work explores a hybrid approach to Tamil hate speech detection by combining target-anchored supervised contrastive learning with lexical feature-based classification. The proposed system aims to improve the detection of implicit hate speech, which is often difficult to identify using conventional keyword-based or standard classification approaches.

---

## Table of Contents

- [Team](#team)
- [Problem Statement](#problem-statement)
- [Proposed Methodology](#proposed-methodology)
- [Experimental Results](#experimental-results)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Technologies and Dependencies](#technologies-and-dependencies)
- [Installation](#installation)
- [Reproducing Experiments](#reproducing-experiments)
- [Limitations and Fairness Analysis](#limitations-and-fairness-analysis)
- [AI Assistance Disclosure](#ai-assistance-disclosure)


---

## Team

**Team Name:** Guns and Roses

- Mohnish Kanna K S
- Oviya T S
- Adithiyaa T
- Sarveash Koushik R V

---

## Problem Statement

Hate speech detection in Tamil social media presents several challenges due to linguistic diversity, code-mixing, transliteration, dialectal variations, and the frequent use of implicit expressions.

Unlike explicit hate speech, implicit hate speech may not contain recognizable slurs or profane language. Instead, it can communicate hostility through context, stereotypes, insinuations, or references to specific target groups.

Conventional lexical classifiers often struggle with such expressions because they rely heavily on surface-level vocabulary.

Our approach investigates whether learning representations anchored around the **target group of hateful expressions**, rather than their surface wording alone, can improve implicit hate speech detection.

---

## Proposed Methodology

The proposed system consists of four major components.

### 1. MuRIL with Target-Anchored Supervised Contrastive Learning

We use Google's `google/muril-base-cased` as the pretrained Transformer backbone.

The model is fine-tuned using a target-anchored supervised contrastive learning objective.

The objective encourages hateful examples associated with the same target group to develop similar representations, irrespective of whether the expressions are explicit or implicit.

This encourages the model to capture target-related semantic patterns rather than relying exclusively on recognizable hateful keywords.

The architecture also incorporates classification heads for the hierarchical prediction tasks.

### 2. Lexical Feature-Based Classification

To complement contextual representations learned by MuRIL, we introduce a lexical classification pipeline using:

- Character-level TF-IDF features.
- Word-level TF-IDF features.
- LightGBM classification.

This component aims to capture lexical variations, slang, transliterated expressions, and dialectal spellings that may not be adequately represented by the Transformer model.

### 3. Probability Calibration and Ensemble Learning

The predictions from the Transformer and lexical models are combined through a calibrated ensemble pipeline.

The process involves:

- **Isotonic Regression:** Calibrates model probability estimates.
- **Logistic Regression Stacking:** Combines predictions from the individual models through a meta-classifier.
- **Threshold Optimization:** Selects decision thresholds based on validation performance.

The final system combines contextual and lexical information to improve classification performance.

### 4. Three-Tier Moderation Triage

Instead of producing only binary predictions, the system incorporates a three-tier decision mechanism:

- **Auto-Flag:** Automatically flags likely hateful content.
- **Auto-Clear:** Automatically clears likely non-hateful content.
- **Human Review:** Sends uncertain predictions for manual assessment.

This approach is intended to support practical content moderation workflows where uncertain cases should not necessarily receive automatic decisions.

---

## Experimental Results

### Level-1 Classification Results

The following table summarizes the experimental configurations evaluated during development.

| Configuration | Accuracy | HATE F1 | NON-HATE F1 | Implicit Recall |
|---|---:|---:|---:|---:|
| MuRIL Level-1 Baseline | 0.83 | 0.84 | 0.81 | ~0.00 |
| Contrastive MuRIL | 0.79 | 0.78 | 0.80 | 0.68 |
| F1-Optimal Threshold | 0.75 | 0.79 | 0.69 | 0.86 |
| Isotonic Calibration | 0.78 | 0.80 | 0.76 | 0.78 |
| **Final Lexical Ensemble** | **0.79** | **0.81** | **0.77** | **0.78** |

The baseline achieved higher overall accuracy but exhibited a significant weakness in detecting implicit hate speech. Its implicit hate recall was approximately zero.

The contrastive learning approach substantially improved implicit hate detection, demonstrating the importance of target-aware representation learning for this task.

The final ensemble achieved a HATE F1-score of 0.81 and implicit hate recall of 0.78.

### Hierarchical Classification Results

| Level | Classification Task | Metric | Score |
|---|---|---|---:|
| Level 1 | HATE / NON-HATE | Weighted F1 | 0.79 |
| Level 2 | Target Group | Macro F1 | 0.27 |
| Level 3 | EXPLICIT / IMPLICIT | Macro F1 | 0.70 |
| Level 5 | Bias Category | Macro F1 | 0.15 |

**Note:** Level 3 evaluation considers only true-HATE examples. Level 4 was not modeled separately due to extreme class sparsity.

The target-group and bias-category classification tasks remain challenging because of limited training examples and class imbalance.

---

## Dataset

The experiments use the Tamil social-media hate speech dataset provided for the DravidianCodeMix 2026 Shared Task.

**Dataset statistics:**

- Total cleaned sentences: 4,246
- Language: Tamil
- Domain: Social media
- Train/Validation/Test split: 70% / 15% / 15%
- Split strategy: Stratified using Level-1 and Level-3 labels

The dataset follows a five-level hierarchical annotation structure:

1. **Level 1:** HATE / NON-HATE
2. **Level 2:** Target Group
3. **Level 3:** EXPLICIT / IMPLICIT
4. **Level 4:** Fine-Grained Subcategory
5. **Level 5:** Bias Category

The dataset is not redistributed in this repository. Researchers should obtain it through the official DravidianCodeMix 2026 Shared Task release.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_baseline_diagnostic.ipynb
│   ├── 03_contrastive_training.ipynb
│   ├── 04_lexical_ensemble.ipynb
│   ├── 05_calibration_and_stacking.ipynb
│   ├── 06_triage_and_error_analysis.ipynb
│   └── 07_fairness_check.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── ensemble.py
│   └── triage.py
│
├── artifacts/
│   └── final_artifacts_v2.pkl
│
├── paper/
│   └── working_notes_paper.docx
│
├── requirements.txt
└── README.md
```

**Notebook descriptions:**

| Notebook | Description |
|---|---|
| `01_data_cleaning.ipynb` | Dataset cleaning and preprocessing |
| `02_baseline_diagnostic.ipynb` | Baseline MuRIL model and diagnostic evaluation |
| `03_contrastive_training.ipynb` | Target-anchored contrastive learning and model training |
| `04_lexical_ensemble.ipynb` | TF-IDF feature extraction and LightGBM training |
| `05_calibration_and_stacking.ipynb` | Probability calibration and ensemble stacking |
| `06_triage_and_error_analysis.ipynb` | Three-tier moderation and error analysis |
| `07_fairness_check.ipynb` | False-positive analysis across keyword groups |

---

## Technologies and Dependencies

The project uses the following technologies:

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Transformer Models | Hugging Face Transformers, MuRIL |
| Machine Learning | Scikit-learn, LightGBM |
| Feature Extraction | TF-IDF |
| Data Processing | Pandas, NumPy |
| Experimentation | Jupyter Notebook |

**Key dependencies:**

```text
torch
transformers
lightgbm
scikit-learn
pandas
numpy
jupyter
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mohnish1234-git/pdfhtml.git
cd pdfhtml
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU-based training, ensure that the installed PyTorch version is compatible with your CUDA environment.

---

## Reproducing Experiments

The notebook - Trial5-NLLB contains the full code.
---

## Limitations and Fairness Analysis

Although the proposed approach improves implicit hate detection, several limitations were identified.

### 1. Target Group Classification

The Level-2 target-group classification task achieved a macro F1-score of 0.27, indicating difficulty in distinguishing between target categories.

### 2. Bias Category Classification

The Level-5 bias category task achieved a macro F1-score of 0.15. Limited examples and class sparsity significantly affect fine-grained classification.

### 3. Fairness Evaluation

We examined false-positive rates among true NON-HATE examples based on the presence of target-group vocabulary.

| Group | False Positive Rate | Sample Count |
|---|---:|---:|
| Contains target-group vocabulary | 0.620 | 50 |
| Does not contain target-group vocabulary | 0.488 | 252 |

The observed difference highlights a potential bias in predictions involving target-group vocabulary.

This finding emphasizes the importance of evaluating hate speech detection systems beyond aggregate classification metrics.
We also didnt get the results of our model's performance on the test data.
---

## AI Assistance Disclosure

In accordance with the DravidianCodeMix 2026 Shared Task AI assistance policy, Claude (Anthropic) was used as a code-writing assistant for the model architecture, training pipeline, and evaluation scripts.

It was also consulted during research framing and experimental design, including the diagnosis of implicit hate detection failures and the development of the target-anchored contrastive learning objective.

All reported results were independently generated by executing the resulting code on the dataset described in the paper.

The analysis, interpretation, and final writing are the authors' own.



**Guns and Roses @ DravidianCodeMix 2026**

*Exploring target-aware representation learning for more robust Tamil hate speech detection.*
