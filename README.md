# Teacher Feedback Classification — NLP System 🎓

> An NLP pipeline that automatically classifies student feedback about teachers into 5 structured categories — built with 6 different models, 4 data augmentation techniques, and a final soft-voting ensemble that achieves **99% accuracy**.

<p>
  <img src="https://img.shields.io/badge/Accuracy-99%25-2E8B57?style=flat-square"/>
  <img src="https://img.shields.io/badge/Models_Tried-6-5C2D91?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black"/>
  <img src="https://img.shields.io/badge/Status-Complete-2E8B57?style=flat-square"/>
</p>

---

## The Problem

Student feedback collected at universities is often unstructured — hundreds of free-text responses that no one has time to manually read and categorise. This system takes raw feedback and automatically routes it into one of five actionable categories, enabling departments to quickly identify whether complaints relate to teaching methods, timing, materials, assessments, or assignments.

---

## 5 Feedback Categories

| Category | Example Feedback |
|---|---|
| **Study Material** | "The notes are uploaded too late for us to prepare." |
| **Lecture Timing** | "Class consistently starts 15 minutes after the scheduled time." |
| **Assessments** | "Quizzes are announced without any prior warning." |
| **Assignments** | "Deadlines are too tight — we don't have enough time to finish." |
| **Study Method** | "Teacher uses real-life examples which makes concepts easy to understand." |

---

## Models Trained & Compared

This project did not settle on the first model that worked. **6 different architectures** were built, evaluated, and compared before selecting the best performer.

| # | Model | Architecture Type | Notes |
|---|---|---|---|
| 1 | **SBERT + SVM (RBF)** | Transformer embedding + kernel SVM | Strong baseline; SVM handles dense 768-dim vectors well |
| 2 | **SBERT + Logistic Regression** | Transformer embedding + linear classifier | Fast, interpretable, competitive |
| 3 | **SBERT + MLP (512→256→128)** | Transformer embedding + neural network | 3-layer MLP with early stopping |
| 4 | **DistilBERT Fine-Tuned** | End-to-end Transformer | State-of-the-art; fine-tuned for 15 epochs with learning rate scheduler |
| 5 | **Stacking Ensemble** | Meta-learning (SVM+LR+MLP → LR) | 5-fold CV stacking; level-1 meta-learner |
| 6 | **BERT + TF-IDF Fusion + SVM** | Hybrid (semantic + lexical) | 768-dim SBERT + word n-grams + char n-grams concatenated |

The best model was automatically selected at runtime and used for the final soft-voting ensemble across all 6 models.

---

## System Architecture

```
Raw Feedback Text
       │
       ├─── preprocess_classical() ──► Stopword removal + Lemmatization
       │                                       │
       │                               TF-IDF (word + char n-grams)
       │
       └─── preprocess_bert() ──────► Light clean (whitespace only)
                                               │
                               SBERT Encoder (all-mpnet-base-v2)
                               768-dimensional semantic embeddings
                                               │
              ┌────────────────┬──────────────┼──────────────┬────────────────┐
              │                │              │              │                │
           SVM (RBF)     Log Reg         MLP 3-layer   DistilBERT     Stacking
              │                │              │          Fine-Tuned    Ensemble
              │                │              │              │                │
              └────────────────┴──────────────┴──────────────┴────────────────┘
                                               │
                              Soft Voting (average probabilities)
                                               │
                                    Final Category + Confidence
```

---

## Data Augmentation — 4 Techniques

To expand the training set and improve generalisation, 4 augmentation methods were applied, growing the dataset to **5× its original size**:

| Technique | What it does | Example |
|---|---|---|
| **Synonym Swap** | Replaces up to 2 words with domain synonyms | "teacher" → "instructor", "quiz" → "assessment" |
| **Word Insertion** | Inserts a filler adverb at a random position | "notes are helpful" → "notes are really helpful" |
| **Random Deletion** | Drops words with 10% probability | "teacher always starts late" → "teacher starts late" |
| **Combined** | Synonym swap + word insertion together | Most diverse augmentation |

---

## Key Technical Decisions

**Why two preprocessing pipelines?**
Classical models (SVM, LR) benefit from aggressive preprocessing — stopword removal, lemmatization, punctuation stripping — because TF-IDF treats every token independently. Transformer models (BERT) handle tokenization internally and actually *lose* information when aggressively cleaned. Keeping both pipelines ensured each model received input in its optimal format.

**Why SBERT over raw BERT for embeddings?**
`all-mpnet-base-v2` is specifically trained for semantic similarity via contrastive learning. "Teacher starts late" and "Class begins delayed" produce similar embeddings — a critical property for feedback classification where paraphrasing is common. Raw BERT's [CLS] token was less effective for this task.

**Why fine-tune DistilBERT separately?**
DistilBERT with a classification head, trained end-to-end on the domain data, allows the attention layers to specialise for academic feedback vocabulary — something pre-computed SBERT embeddings cannot do.

**Why a stacking ensemble on top of individual models?**
Stacking learns *which* models to trust for *which* input patterns, rather than trusting each model equally. The meta-learner (Logistic Regression) was trained on out-of-fold predictions via 5-fold cross-validation to prevent data leakage.

---

## Results

```
MODEL COMPARISON
═══════════════════════════════════════════════════
  SBERT + SVM              :  ██████████████████████████
  SBERT + Logistic Reg     :  ██████████████████████████
  SBERT + MLP (512-256-128):  ██████████████████████████
  DistilBERT Fine-Tuned    :  ██████████████████████████
  Stacking Ensemble        :  ██████████████████████████
  BERT + TF-IDF Fusion     :  ██████████████████████████

🏆 Final Accuracy : 99%   (Soft-vote ensemble — all 6 models)

5-Fold Cross-Validation (SBERT + SVM):
  Mean accuracy : confirms result stability
  Std deviation : < 1%
═══════════════════════════════════════════════════
```

---

## Repository Structure

```
teacher-feedback-nlp/
│
├── teacher_feedback_nlp_v3_FINAL.ipynb   ← Main notebook (Section A: Train + Section B: Predict)
│
├── dataset/
│   └── teacher_feedback_dataset.csv      ← Labelled feedback dataset (5 categories)
│
├── models/                               ← Saved after running Section A
│   ├── teacher_feedback_v3_bundle.pkl    ← All sklearn models + vectorizers + label encoder
│   └── teacher_feedback_distilbert/      ← DistilBERT weights (HuggingFace format)
│
├── outputs/
│   └── teacher_feedback_v3_predictions.csv  ← Per-sample prediction report
│
└── README.md
```

---

## How to Run

### Option A — Google Colab (recommended)

1. Open `teacher_feedback_nlp_v3_FINAL.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change Runtime Type → T4 GPU**
3. Upload `teacher_feedback_dataset.csv` to your Google Drive
4. Update `dataset_path` in Step 3 if needed
5. Run **Section A** (cells top to bottom) — trains all 6 models and saves to Drive
6. Run **Section B** independently in any session — loads saved models and predicts

### Option B — Local (CPU, slower for DistilBERT)

```bash
git clone https://github.com/SadiaIlyas/teacher-feedback-nlp.git
cd teacher-feedback-nlp

pip install sentence-transformers transformers torch scikit-learn pandas numpy matplotlib seaborn nltk
```

Then open the notebook in Jupyter and comment out the `drive.mount()` lines, replacing paths with local paths.

---

## Prediction Example

```python
MY_REVIEW = "The teacher is very good at explaining complex topics simply."

result = predict_all_models(MY_REVIEW)

# Output:
# Category    : Study Method
# Confidence  : 98.74%
#
# Individual Model Votes:
#   SBERT+SVM       : Study Method
#   SBERT+LR        : Study Method
#   SBERT+MLP       : Study Method
#   DistilBERT      : Study Method
#   Stacking        : Study Method
#   BERT+TF-IDF     : Study Method
#
# Probability per Category:
#   Study Method          :  98.74%  █████████████████████████████████
#   Lecture Timing        :   0.52%
#   Study Material        :   0.39%
#   Assessments           :   0.22%
#   Assignments           :   0.13%
```

---

## Libraries & Tools

| Library | Purpose |
|---|---|
| `sentence-transformers` | SBERT embeddings (`all-mpnet-base-v2`) |
| `transformers` (HuggingFace) | DistilBERT tokenizer + sequence classification model |
| `torch` (PyTorch) | DistilBERT training loop, GPU acceleration |
| `scikit-learn` | SVM, Logistic Regression, MLP, Stacking, TF-IDF, metrics |
| `nltk` | Stopword removal, lemmatization (WordNetLemmatizer) |
| `pandas` / `numpy` | Data manipulation and array operations |
| `matplotlib` / `seaborn` | Training curves, confusion matrix, bar charts |
| `pickle` | Model serialisation |

---

## What I Learned

This project did not produce 99% on the first attempt. It took **6 model iterations** to get there:

- **Attempt 1** — TF-IDF + Naive Bayes: fast but semantically blind. Synonyms and paraphrases were misclassified.
- **Attempt 2** — TF-IDF + SVM: better, but still no semantic understanding.
- **Attempts 3–4** — SBERT embeddings with SVM, LR, and MLP: significant jump. Semantic similarity finally captured.
- **Attempt 5** — DistilBERT end-to-end fine-tuning: learned domain-specific patterns the pre-trained embeddings missed.
- **Attempt 6** — Fusing SBERT with TF-IDF: combining semantic and lexical signals for the hybrid model.
- **Final** — Soft-voting all 6 models: ensemble disagreement identified the hardest edge cases and resolved them.

The lesson: in NLP, no single model is optimal for all input patterns. Building and comparing multiple architectures — then combining their predictions — is standard practice in production systems.

---

## Author

**Sadia Ilyas** — 0078-BSCS-24 | CS Student @ GCU Lahore
[LinkedIn](https://linkedin.com/in/sadia-ilyas-b96183353) · [GitHub](https://github.com/SadiaIlyas)
