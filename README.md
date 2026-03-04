# Indonesian Sociolinguistics Text Classifier
### NLP Week 5 — Sociolinguistics

## Project Overview
A text classification system that categorizes Indonesian text into **3 sociolinguistic varieties**:

| Class | Description | Source |
|-------|-------------|--------|
| **Alay/Slang** | Informal Indonesian internet slang (e.g., "bgt", "gk", "kk") | [kamus-alay](https://github.com/nasalsabila/kamus-alay) |
| **Jaksel (Indonglish)** | Indonesian-English code-switching ("literally gue tuh...") | `dataset.csv` |
| **Bahasa Baku (EYD)** | Formal Indonesian following EYD rules | Derived from formal corrections + template |

## Project Structure
```
├── colloquial-indonesian-lexicon.csv  # Raw alay/slang data
├── dataset.csv                        # Raw jaksel data
├── clean_alay.py                      # Alay data cleaning script
├── clean_jaksel.py                    # Jaksel data cleaning script
├── alay_cleaned.csv                   # Cleaned alay dataset (4,877 rows)
├── jaksel_cleaned.csv                 # Cleaned jaksel dataset (5,065 rows)
├── slang_dictionary.csv               # Slang→Formal word mapping
├── alay_categories.csv                # Slang category metadata
├── eyd_template.csv                   # Blank EYD template (to be filled)
├── eda.py                             # Exploratory Data Analysis script
├── eda_output/                        # EDA visualization outputs
├── model.py                           # Classification model training
├── sociolinguistics_model.pkl         # Trained model (best: LR, 90.89%)
└── README.md                          # This file
```

## Progress

- [x] Dataset collection (Alay + Jaksel)
- [x] Data cleaning & preprocessing
- [x] Exploratory Data Analysis (EDA)
- [x] Model training & evaluation
- [ ] **EYD dataset integration** (template ready, data TBD)
- [ ] Model retraining with EYD data

## EDA Findings

| Metric | Alay/Slang | Jaksel |
|--------|-----------|--------|
| Samples | 4,877 | 5,065 |
| Avg text length | 72.2 chars | 105.4 chars |
| Avg word count | 12.4 | 17.9 |
| Top slang category | abreviasi (7,162) | — |
| English word ratio | — | 24.4% avg |

## Model Results

Best model: **Logistic Regression** with char n-gram TF-IDF (2-5 grams)

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **90.89%** ★ |
| Linear SVM | 90.43% |
| Multinomial NB | 83.96% |
| Word-level LR | 87.78% |

### Per-class Performance (Best Model)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Alay | 0.90 | 0.83 | 0.86 |
| Baku | 0.85 | 0.92 | 0.88 |
| Jaksel | 0.98 | 0.97 | 0.98 |

## How to Run

```bash
# 1. Clean datasets
python clean_alay.py
python clean_jaksel.py

# 2. Run EDA
python eda.py

# 3. Train model
python model.py
```

## TODO
- [ ] Add EYD (Ejaan Yang Disempurnakan) dataset with proper formal Indonesian text
- [ ] Retrain model with balanced 3-class data including real EYD samples
- [ ] Experiment with transformer-based models (IndoBERT)
