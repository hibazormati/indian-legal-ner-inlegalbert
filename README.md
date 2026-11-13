
# Legal Named Entity Recognition
### Information Extraction from Indian Court Decisions using InLegalBERT

<p align="center">
  <b>Hiba Zormati</b> · Statistics & Data Science Engineering · ESSAI Tunis
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-Named_Entity_Recognition-16A085" />
  <img src="https://img.shields.io/badge/Transformer-InLegalBERT-6366F1" />
  <img src="https://img.shields.io/badge/Baseline-CRF-orange" />
  <img src="https://img.shields.io/badge/Framework-Hugging_Face-yellow?logo=huggingface" />
</p>

---

## Overview

Legal documents contain large amounts of structured information hidden within lengthy and complex textual content.

Identifying judges, lawyers, parties, courts, statutes, provisions and legal precedents is essential for building intelligent legal information systems, improving case-law search and supporting automated document analysis.

This project develops a **Named Entity Recognition (NER) pipeline for Indian court decisions**, using the InLegalNER dataset.

Two approaches are implemented and compared:

- **Conditional Random Fields (CRF):** A classical sequence-labeling baseline using handcrafted linguistic features.
- **InLegalBERT:** A pretrained language model fine-tuned for legal entity extraction.

The project also investigates an important methodological issue in Transformer-based NER: the impact of the 512-subword input limit on long legal documents and the evaluation bias introduced by truncation.

**Research question:** How much does fine-tuned InLegalBERT improve the recognition of legal entity types compared with a feature-engineered CRF baseline, and how can long-document truncation affect evaluation validity?

## Dataset

The project uses **InLegalNER**, a dataset of Indian court judgments containing fine-grained legal entity annotations.

- Dataset: [InLegalNER – Hugging Face](https://huggingface.co/datasets/opennyaiorg/InLegalNER)
- Source of legal texts: Indian Kanoon
- Annotation format: Label Studio JSON with character-level offsets.
- Task: Multi-class Named Entity Recognition.

### Dataset splits

| Split | Examples |
|---|---:|
| Training | 10,995 |
| Validation | 1,074 |
| Testing | 4,501 |

### Entity taxonomy

The dataset defines 14 entity categories:

| Category | Description |
|---|---|
| CASE_NUMBER | Legal case identifiers |
| COURT | Courts and judicial institutions |
| DATE | Dates |
| GPE | Geopolitical entities |
| JUDGE | Judges |
| LAWYER | Legal representatives |
| ORG | Organizations |
| OTHER_PERSON | Other persons |
| PETITIONER | Petitioners |
| PRECEDENT | Legal precedents |
| PROVISION | Legal provisions |
| RESPONDENT | Respondents |
| STATUTE | Laws and statutes |
| WITNESS | Witnesses |

## Methodology

### 1. Data exploration and annotation audit

Before model training, the dataset was examined to verify annotation consistency and understand the challenges of legal text processing.

The audit included:

- Dataset split inspection and label distribution.
- Character-offset validation.
- Detection of overlapping entity spans.
- Analysis of punctuation within annotated entities.
- Examination of legal abbreviations and tokenization issues.
- Analysis of entity alignment with word-level tokens.

The annotation audit identified no invalid character spans or overlapping entities in the training split.

### 2. Text preprocessing and BIO encoding

Legal text requires careful tokenization because entity boundaries can occur around punctuation and legal abbreviations.

A word-and-punctuation tokenizer was implemented using:

` \w+|[^\w\s] `

Character-level annotations were converted into the BIO tagging scheme:

- **B-ENTITY:** Beginning of an entity.
- **I-ENTITY:** Continuation of an entity.
- **O:** Outside any entity.

This representation supports token-level sequence labeling while preserving the original entity categories.

### 3. CRF baseline

A Conditional Random Field model was implemented using handcrafted linguistic features.

Features include:

- Word identity and capitalization.
- Prefixes and suffixes.
- Digit and punctuation patterns.
- Contextual neighboring words.
- Position-related linguistic information.

The CRF provides a classical baseline for evaluating the contribution of contextual Transformer representations.

### 4. InLegalBERT fine-tuning

The advanced model uses [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), a pretrained language model designed for Indian legal text.

The model was adapted for token classification using:

- Hugging Face Transformers.
- BIO token-label alignment.
- A token classification head.
- Dynamic batch padding.
- Validation-based model selection.

**Training configuration**

| Parameter | Value |
|---|---:|
| Learning rate | 2e-5 |
| Training batch size | 16 |
| Evaluation batch size | 16 |
| Epochs | 4 |
| Weight decay | 0.01 |
| Maximum sequence length | 512 subwords |
| Random seed | Configured in notebook |

The best checkpoint was selected using entity-level F1 on the validation set.

## Long-Document Evaluation: The 512-Token Problem

A significant methodological issue emerged during the initial Transformer evaluation.

Standard BERT inference processes only one sequence of up to 512 subwords per document. For long court decisions, this silently excludes entities appearing beyond the model's input window.

The initial evaluation omitted **704 of the 13,357 word-level gold test entities** (approximately 5.3%).

This made the initial Transformer evaluation incomplete and unsuitable for a direct comparison with the CRF, which processed the full documents.

### Sliding-window inference

To address this issue, a sliding-window inference strategy was implemented.

Instead of processing only the beginning of each document, the text is divided into successive overlapping windows, allowing predictions to cover the full document.

```text
Long Court Decision
         |
         v
  Tokenized Document
         |
         v
 +------------------+
 | Window 1         |
 +------------------+
       overlap
 +------------------+
 | Window 2         |
 +------------------+
       overlap
 +------------------+
 | Window 3         |
 +------------------+
         |
         v
 Predictions across
 the full document
         |
         v
 Reconstructed NER
 predictions
```

The corrected test evaluation uses sliding-window inference for InLegalBERT and full-document predictions for CRF, ensuring that both models are evaluated against the same 13,357 gold entities.

## Experimental Results

Both models are evaluated using entity-level precision, recall and F1-score.

An entity is considered correctly identified only when its type and both boundaries match the gold annotation.

### Overall performance

| Metric | CRF | InLegalBERT |
|---|---:|---:|
| Micro Precision | 0.705 | 0.816 |
| Micro Recall | 0.569 | 0.884 |
| Micro-F1 | 0.6296 | **0.8483** |
| Macro-F1 | 0.5921 | **0.8373** |

**Performance improvement**

- Micro-F1: +0.2187
- Macro-F1: +0.2453

### Key observations

**1. Contextual representations improve role recognition**

InLegalBERT shows its largest improvements on role-dependent entities, including WITNESS, PETITIONER and RESPONDENT.

These categories require understanding the legal context rather than relying exclusively on the surface form of a person's name.

**2. Stronger recall**

InLegalBERT achieves a micro-recall of 0.884 compared with 0.569 for CRF, indicating that it identifies a larger proportion of the gold entities.

**3. Remaining classification challenges**

PRECEDENT, ORG and GPE remain among the weakest InLegalBERT categories.

Complex legal citations, heterogeneous organization names and exact entity boundaries remain challenging.

**4. Evaluation methodology matters**

The initial 512-subword evaluation excluded entities from long documents. Sliding-window inference corrected this test-time coverage problem and enabled a more valid comparison.

## Limitations

Despite the improvement over the baseline, several limitations remain:

- Only one random seed and one hyperparameter configuration were used.
- No generic BERT control was trained, so the specific contribution of legal-domain pretraining cannot be isolated.
- Training and validation still use truncated sequences; sliding windows were introduced for test-time inference only.
- The CRF baseline was not extensively hyperparameter-tuned.
- The first-subword labeling strategy does not enforce valid BIO transitions.
- Nine entities could not be aligned with whole-word tokens and were excluded (eight from the test set).
- Evaluation relies on exact entity boundaries.
- The dataset represents Indian legal documents, limiting generalization to other jurisdictions.

## Future Research Directions

Potential improvements include:

- Training with sliding-window or overflow preprocessing.
- Adding a generic BERT baseline to isolate domain-specific pretraining effects.
- Integrating a CRF layer on top of BERT for structured BIO predictions.
- Exploring long-context Transformer architectures.
- Incorporating document-zone information such as preamble and judgment sections.
- Testing multiple random seeds and statistical significance.
- Investigating few-shot LLM-based legal information extraction.

## Technologies

| Category | Tools |
|---|---|
| Programming | Python |
| NLP | Tokenization, BIO Tagging, NER |
| Classical ML | Conditional Random Fields |
| Transformers | InLegalBERT, BERT |
| Deep Learning | PyTorch |
| NLP Frameworks | Hugging Face Transformers, Datasets |
| Evaluation | Seqeval, Scikit-learn |
| Environment | Google Colab, NVIDIA T4 GPU |

## References

- Kalamkar et al. (2022). *Named Entity Recognition in Indian Court Judgments*. Natural Legal Language Processing Workshop.
- Paul et al. (2023). *Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law*.
- Lafferty et al. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Tjong Kim Sang & De Meulder (2003). *Introduction to the CoNLL-2003 Shared Task*.

**Dataset:** [InLegalNER](https://huggingface.co/datasets/opennyaiorg/InLegalNER)

**Pretrained model:** [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)

---

## Disclaimer

This project is an academic NLP experiment for legal information extraction. Its predictions are not legal advice and should not be treated as authoritative legal interpretations.

---

<p align="center">
  <b>Developed by Hiba Zormati</b><br/>
  Statistics & Data Science Engineering · ESSAI Tunis<br/>
  <a href="https://www.linkedin.com/in/hiba-zormati-624b13319/">LinkedIn</a> ·
  <a href="https://github.com/hibazormati">GitHub</a>
</p>
