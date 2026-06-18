# Transformer Fine-Tuning for Response Clarity Classification (QEvasion)

Fine-tuning and comparison of three transformer encoders — **BERT**, **DistilBERT**, and **DeBERTa-v3** — for **response clarity classification** on the **SemEval 2026 Task 6 (QEvasion)** dataset. Given a political interview **question** and the politician's **answer**, the task is to classify how directly the answer responds (e.g. *Clear Reply*, *Ambivalent*, *Clear Non-Reply*).

> Coursework project for **Artificial Intelligence II — Deep Learning for NLP** (BSc in Informatics & Telecommunications, NKUA), submitted to SemEval 2026 Task 6.

## Problem

Interview answers are often evasive in subtle ways. The goal is to train a classifier that scores the *clarity* of a response relative to the question it was given — a 3-way classification problem with strong class imbalance (the *Ambivalent* class dominates).

## Approach

The question and answer are passed as **two separate sequences** so the model gets an explicit Q/A boundary via segment embeddings:

```
[CLS] <question> [SEP] <answer> [SEP]
```

Each model is fine-tuned with the same training pipeline for a fair comparison:

- **Class-weighted loss** (balanced) to counter label imbalance
- **Early stopping** on validation Macro-F1 (patience = 2)
- **Best-checkpoint reload** before inference — final-epoch weights are *not* used
- Decoupled AdamW, warmup (10%), gradient clipping (1.0), max seq length 256

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Epochs | 5 + early stopping |
| Batch size | 16 |
| Max sequence length | 256 |
| Warmup ratio | 10% |
| Weight decay | 0.01 |
| Class weights | balanced |

## Results

Validation Macro-F1 (best checkpoint per model):

| Model | Val Accuracy | Val Macro-F1 |
|---|---|---|
| **BERT-base** | 0.69 | **0.663** |
| **DistilBERT-base** | 0.68 | 0.658 |
| **DeBERTa-v3-base** | 0.66 | 0.589 |

**BERT-base** gave the best clarity classification, with **DistilBERT** within ~0.5 points at a fraction of the parameters — a strong speed/accuracy trade-off.

## Key findings

- **DistilBERT is the practical winner.** It reaches 0.658 Macro-F1 vs BERT's 0.663 while being ~40% smaller and faster — for this task the distilled model loses almost nothing.
- **DeBERTa-v3 underperformed and trained unstably.** Despite being the strongest model on paper, it climbed slowly and erratically (0.39 -> 0.59 across epochs) and never matched the base BERT models. On a dataset this size, the larger model's capacity did not pay off.
- **Best-checkpoint reload matters.** Validation Macro-F1 peaks mid-training (e.g. BERT at epoch 4) and then degrades as the model overfits — always restore the best checkpoint before inference rather than using the last epoch.
- **Class weighting was essential.** Without it, models collapsed toward the majority *Ambivalent* class.

## Error analysis

The `deberta` notebook includes a misclassification breakdown. The dominant error mode is *Ambivalent -> Clear Reply*: answers that engage with the topic but never actually commit, which the model reads as direct replies.

## Repo structure

```
.
├── notebooks/
│   ├── bert.ipynb        # BERT-base-uncased
│   ├── distilbert.ipynb  # DistilBERT-base-uncased
│   └── deberta.ipynb     # DeBERTa-v3-base
├── requirements.txt
└── README.md
```

## Running

```bash
pip install -r requirements.txt
jupyter notebook
```

The QEvasion dataset is from SemEval 2026 Task 6 and is not redistributed here.

## Stack

PyTorch · Hugging Face Transformers · scikit-learn · pandas
