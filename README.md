# Indonesian Sociolinguistics Text Classifier
### NLP Week 5 — Sociolinguistics

## Project Overview
A text classification system that categorizes Indonesian text into **3 sociolinguistic varieties**:

| Class | Description | Source |
|-------|-------------|--------|
| **EYD (Formal)** | Formal Indonesian following Ejaan Yang Disempurnakan rules | [Tere Liye — Tentang Kamu](https://archive.org/stream/tentangkamu/TENTANG%20KAMU_djvu.txt) |
| **Alay/Slang** | Informal Indonesian internet slang (e.g., "bgt", "gk", "kk") | [kamus-alay](https://github.com/nasalsabila/kamus-alay) |
| **Jaksel (Indonglish)** | Indonesian-English code-switching ("literally gue tuh...") | [indonglish-dataset](https://github.com/laksmitawidya/indonglish-dataset) |

## Project Structure
```
├── source-data/                           # Raw source datasets
│   ├── colloquial-indonesian-lexicon.csv  #   Alay/slang lexicon
│   ├── dataset.csv                        #   Jaksel tweets
│   └── eyd_tere_liye_tentang_kamu.txt     #   Novel text (EYD source)
├── cleaned-data/                          # Cleaned datasets
│   ├── eyd_cleaned.csv                    #   Cleaned EYD (9,118 rows)
│   ├── alay_cleaned.csv                   #   Cleaned alay (4,877 rows)
│   ├── jaksel_cleaned.csv                 #   Cleaned jaksel (5,065 rows)
│   ├── slang_dictionary.csv               #   Slang→Formal word mapping
│   └── alay_categories.csv               #   Slang category metadata
├── clean_eyd.py                           # EYD data cleaning script
├── clean_alay.py                          # Alay data cleaning script
├── clean_jaksel.py                        # Jaksel data cleaning script
├── eda.py                                 # Exploratory Data Analysis script
├── eda_output/                            # EDA visualization outputs
├── model.py                               # Classification model training
├── sociolinguistics_model.pkl             # Trained model (best: SVM, 99.00%)
├── ch06.ipynb                             # Reference notebook (Ch6 finetuning)
└── README.md                              # This file
```

## Progress

- [x] Dataset collection (EYD + Alay + Jaksel)
- [x] Data cleaning & preprocessing
- [x] Exploratory Data Analysis (EDA)
- [x] Model training & evaluation

## EDA Findings

| Metric | EYD (Formal) | Alay/Slang | Jaksel |
|--------|-------------|-----------|--------|
| Samples | 9,118 | 4,877 | 5,065 |
| Avg text length | ~70 chars | ~72 chars | ~105 chars |
| Avg word count | ~11 | ~12 | ~18 |
| Top slang category | — | abreviasi (7,162) | — |
| English word ratio | Low | Low | ~24.4% avg |

## Model Results

Best model: **Linear SVM** with char n-gram TF-IDF (2-5 grams)

| Model | Accuracy |
|-------|----------|
| **Linear SVM** | **99.00%** ★ |
| Logistic Regression | 98.01% |
| Multinomial NB | 94.64% |
| Word-level LR | 96.39% |

### Per-class Performance (Best Model)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Alay | 0.9896 | 0.9825 | 0.9860 |
| EYD | 0.9887 | 0.9949 | 0.9918 |
| Jaksel | 0.9918 | 0.9928 | 0.9923 |

## How to Run

```bash
# 1. Clean datasets
python clean_eyd.py
python clean_alay.py
python clean_jaksel.py

# 2. Run EDA
python eda.py

# 3. Train model
python model.py
```
