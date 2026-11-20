# Tamil Hate Speech Detection | DravidianCodeMix 2026

### A Hierarchical Multi-Task MuRIL Framework with Back-Translation, Out-of-Fold Hard-Negative Mining and F2-Tuned Thresholds

This repository contains the Jupyter Notebook implementation and experimental work developed for our participation in the **DravidianCodeMix 2026 Shared Task on Hate and Hope Speech Detection in Dravidian Languages**.

The project focuses on detecting explicit and implicit hate speech in Tamil social media using a hierarchical multi-task learning framework built on Google's MuRIL Transformer architecture.

The approach combines multilingual Transformer fine-tuning, back-translation-based data augmentation, out-of-fold hard-example mining, class-weighted focal loss, and F2-score-based threshold optimization.

All experiments can be explored and executed using the provided Jupyter Notebook in Google Colab with a T4 GPU runtime.

---

## Team

**Team Name:** Guns and Roses

Team Leader :
- Oviya T S
Team Members : 
- Mohnish Kanna K S
- Adithiyaa T
- Sarveash Koushik R V

---

## Problem Statement

Hate speech detection in Tamil social media is challenging because hateful expressions are frequently conveyed through sarcasm, stereotypes, indirect statements, and contextual references rather than explicit slurs or offensive vocabulary.

The task becomes more difficult due to limited annotated data, code-mixed language, transliterated Tamil, and highly imbalanced fine-grained labels.

The DravidianCodeMix 2026 Tamil track, *Hierarchical Gender-Aware Tamil Hate Speech Detection*, requires classification across five interconnected levels.

| Level | Classification Task |
|---|---|
| Level 1 | HATE / NON-HATE |
| Level 2 | Target Group |
| Level 3 | EXPLICIT / IMPLICIT |
| Level 4 | Fine-grained Hate Subcategory |
| Level 5 | Bias Category |

Our objective was to develop a hierarchical classification system that improves implicit hate detection while maintaining consistency between predictions across different annotation levels.

---

## Proposed Methodology

### 1. Hierarchical Multi-Task MuRIL Architecture

The backbone of the system is Google's pretrained `google/muril-base-cased` model.

MuRIL (Multilingual Representations for Indian Languages) is a BERT-style Transformer pretrained on Indian languages and English, including transliterated text.

The model uses a shared Transformer encoder followed by five task-specific classification heads.

**Architecture details:**

- Pretrained encoder: `google/muril-base-cased`
- Maximum input length: 128 tokens
- Sentence representation: Concatenation of the `[CLS]` representation and mean-pooled token representations.
- Classification heads: Two-layer feed-forward networks.
- Activation: GELU
- Dropout: 0.2

Lower-level classification heads are conditioned on predictions from higher levels. Constrained decoding ensures that the final predictions follow the hierarchical label structure.

### 2. Hierarchical Label Processing and Multi-Task Learning

The dataset contains missing annotations at certain hierarchical levels.

Instead of treating missing annotations as an additional class, per-level loss masks are used to exclude missing labels from the corresponding loss calculations.

The overall training objective combines five classification losses using weighted focal cross-entropy.

Training configurations include:

- Focal loss parameter: γ = 1
- Label smoothing: 0.05
- Inverse-frequency-based class weighting
- Per-level loss weighting
- Separate classification heads for all five levels

### 3. NLLB-200 Back-Translation

To increase training diversity and improve generalization, we use the distilled 600M-parameter NLLB-200 translation model.

Tamil sentences are translated into a pivot language and then translated back into Tamil.

The augmentation strategy uses:

- Tamil → English → Tamil
- Tamil → Hindi → Tamil

Implicit-hate examples and minority-class examples receive additional augmentation.

**Augmentation statistics:**

| Description | Count |
|---|---:|
| Total translation jobs | 3,310 |
| English pivot translations | 2,323 |
| Hindi pivot translations | 987 |
| Retained paraphrases | 2,782 |
| Original training sentences | 2,968 |
| Final augmented training rows | 5,750 |

Filtering is applied to remove unsuitable paraphrases, including duplicates, empty outputs, excessive length differences, insufficient Tamil-script content, and extreme character-level similarity.

Data splitting is performed before augmentation to prevent validation and test data leakage.

### 4. Out-of-Fold Hard-Negative Mining

We incorporate out-of-fold (OOF) prediction-based hard-example mining to identify difficult training examples.

The training dataset is divided into three stratified folds. For each fold, a model is trained on the other two folds and generates predictions for the held-out fold.

The mining strategy identifies:

- **Hard negatives:** NON-HATE examples with high predicted hate probabilities.
- **Hard positives:** HATE examples with low predicted hate probabilities.

Selected hard examples and their paraphrases receive an increased sample weight of 2.0 during the second training stage.

Strong disagreements between model predictions and ground-truth labels are also recorded for manual inspection to identify potential annotation errors.

### 5. F2-Tuned Thresholds and Constrained Decoding

The default classification threshold of 0.5 may not be suitable when recall is particularly important.

We optimize decision thresholds using validation data.

For Level 1, the threshold is selected to maximize the F2-score of the HATE class, giving greater importance to recall than precision.

For Levels 2–5, class-specific offsets are optimized using coordinate ascent to improve macro-F2.

Constrained decoding ensures that predictions remain consistent with the hierarchical label structure.

---

## Dataset

The experiments use the Tamil training dataset released for the DravidianCodeMix 2026 Shared Task.

| Attribute | Details |
|---|---|
| Language | Tamil |
| Domain | Social media |
| Original sentences | 4,255 |
| Cleaned sentences | 4,240 |
| HATE | 2,228 |
| NON-HATE | 2,012 |
| Training samples | 2,968 |
| Validation samples | 636 |
| Test samples | 636 |
| Split | 70% / 15% / 15% |

### Data Preprocessing

The preprocessing pipeline includes:

- Removal of missing Level-1 labels.
- Removal of exact duplicates and conflicting labels.
- Unicode NFC normalization.
- URL removal.
- Replacement of user mentions with placeholders.
- Normalization of excessively repeated characters.
- Merging of extremely sparse fine-grained categories.
- Per-level missing-label handling.

The dataset is stratified using Level-1 labels combined with Level-2 target groups.

---

## Experimental Results

Experiments were conducted using a single NVIDIA T4 GPU through Google Colab with mixed-precision FP16 training.

The model uses AdamW optimization, weight decay, linear learning-rate warm-up, and linear learning-rate decay.

### 1. Level-1 Hate Speech Classification

| Model | Threshold | Macro F1 | Hate Recall | Hate F2 | Implicit Recall |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 0.50 | 0.798 | 0.817 | 0.814 | 0.726 |
| Stage 1 | 0.06 | 0.759 | 0.910 | 0.865 | 0.863 |
| Stage 2 | 0.50 | 0.792 | 0.838 | 0.827 | 0.760 |
| Stage 2 | F2-tuned | 0.771 | 0.895 | 0.858 | 0.851 |

The Stage-1 model at the default threshold achieves:

- Accuracy: 0.80
- Macro F1: 0.80

With F2-based threshold optimization, hate recall increases from approximately 0.82 to 0.91, while implicit hate recall increases from approximately 0.73 to 0.86.

This improvement comes at the cost of increased false positives and reduced overall accuracy.

### 2. Hierarchical Classification Results

| Level | Classification Task | Default Oracle Macro F1 | F2 Oracle Macro F1 | End-to-End Macro F1 |
|---|---|---:|---:|---:|
| Level 2 | Target Group | 0.628 | 0.610 | 0.603 |
| Level 3 | Explicit / Implicit | 0.742 | 0.727 | — |
| Level 4 | Subcategory (24 classes) | 0.172 | 0.161 | 0.152 |
| Level 4 | Abuse Type (16 classes) | 0.198 | 0.194 | — |
| Level 5 | Bias Type | 0.375 | 0.389 | — |

Oracle scores evaluate lower-level classification on true HATE examples, while end-to-end scores also account for errors at Level 1.

### 3. Baseline Comparison

| Level | Metric | Initial Baseline | Proposed System |
|---|---|---:|---:|
| Level 1 | Accuracy | 0.80 | 0.80 |
| Level 2 | Macro F1 | 0.12 | 0.63 |
| Level 3 | Macro F1 | 0.47 | 0.74 |
| Level 4 | Macro F1 | 0.02 | 0.17 |
| Level 5 | Macro F1 | 0.15 | 0.38 |

The hierarchical architecture significantly improves lower-level classification compared with the initial baseline.

---

## Repository Contents

The repository contains the Jupyter Notebook used for data preprocessing, model training, augmentation, evaluation, and experimentation.

The notebook can be downloaded directly from the repository and executed in Google Colab.

---

## How to Run the Project

The experiments are designed to run in **Google Colab with a T4 GPU runtime**.

### Step 1: Download the Notebook

Open the repository and navigate to the Jupyter Notebook (`.ipynb`) containing the implementation.

Download the notebook to your local system.

### Step 2: Open Google Colab

Visit [Google Colab](https://colab.research.google.com/).

Select:

`File → Upload Notebook`

Upload the downloaded `.ipynb` file.

### Step 3: Enable GPU Runtime

In Google Colab:

1. Open Runtime.
2. Select Change runtime type.
3. Set Hardware accelerator to GPU.
4. Select T4 GPU, if available.
5. Save the configuration.

### Step 4: Execute the Notebook

Run the notebook cells sequentially.

The notebook handles the required preprocessing, model initialization, training, evaluation, and experimentation.

Make sure the required dataset files are available at the locations specified in the notebook.

**Note:** Training and NLLB-200 back-translation can take considerable time and GPU resources. Runtime availability and execution time depend on Google Colab's resource limits.

---

## Limitations

Some limitations identified during experimentation include:

- The experiments were conducted using a single random seed.
- The independent contribution of back-translation has not been established through a dedicated ablation.
- The F2-optimized threshold indicates potential probability calibration issues.
- Fine-grained categories suffer from class imbalance and limited training examples.
- Recall-oriented thresholding increases false-positive predictions.
- The dataset contains relatively few entirely romanized Tamil examples.

These limitations highlight opportunities for further experimentation and improvement.

---

## Team

**Guns and Roses | DravidianCodeMix 2026**

Developed as part of our participation in the DravidianCodeMix 2026 Shared Task on Hate and Hope Speech Detection in Dravidian Languages.

