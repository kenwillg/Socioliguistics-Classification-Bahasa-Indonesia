"""
Indonesian Sociolinguistics Text Classifier
3-class classification: EYD (Formal), Alay/Slang, Jaksel (Indonglish)

Strategy:
  - EYD samples: formal Indonesian sentences from Tere Liye's novel "Tentang Kamu"
  - Alay samples: context sentences from colloquial lexicon (contain slang)
  - Jaksel samples: code-switched tweets (Indonesian-English mix)
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_DIR = 'cleaned-data'

# ── Load Data ───────────────────────────────────────────────
print("=" * 60)
print("INDONESIAN SOCIOLINGUISTICS CLASSIFIER")
print("=" * 60)

# Load cleaned datasets
eyd_df = pd.read_csv(os.path.join(DATA_DIR, 'eyd_cleaned.csv'))
alay_df = pd.read_csv(os.path.join(DATA_DIR, 'alay_cleaned.csv'))
jaksel_df = pd.read_csv(os.path.join(DATA_DIR, 'jaksel_cleaned.csv'))

print(f"\nLoaded:")
print(f"  EYD samples:    {len(eyd_df)}")
print(f"  Alay samples:   {len(alay_df)}")
print(f"  Jaksel samples: {len(jaksel_df)}")


# ══════════════════════════════════════════════════════════════
# STEP 1: PREPARE DATASETS
# ══════════════════════════════════════════════════════════════
print("\n--- Preparing datasets ---")

# Ensure consistent columns
eyd_df = eyd_df[['text', 'label']].dropna().drop_duplicates(subset=['text'])
alay_df = alay_df[['text', 'label']].dropna().drop_duplicates(subset=['text'])
jaksel_df = jaksel_df[['text', 'label']].dropna().drop_duplicates(subset=['text'])

# Filter very short texts
eyd_df = eyd_df[eyd_df['text'].str.len() >= 5].reset_index(drop=True)
alay_df = alay_df[alay_df['text'].str.len() >= 5].reset_index(drop=True)
jaksel_df = jaksel_df[jaksel_df['text'].str.len() >= 5].reset_index(drop=True)

print(f"  After cleaning:")
print(f"  EYD:    {len(eyd_df)}")
print(f"  Alay:   {len(alay_df)}")
print(f"  Jaksel: {len(jaksel_df)}")


# ══════════════════════════════════════════════════════════════
# STEP 2: BALANCE AND COMBINE DATASETS
# ══════════════════════════════════════════════════════════════
print("\n--- Combining datasets ---")

# Balance by downsampling to the smallest class size
min_size = min(len(eyd_df), len(alay_df), len(jaksel_df))
print(f"  Balancing to: {min_size} samples per class")

eyd_balanced = eyd_df.sample(n=min_size, random_state=RANDOM_STATE)
alay_balanced = alay_df.sample(n=min_size, random_state=RANDOM_STATE)
jaksel_balanced = jaksel_df.sample(n=min_size, random_state=RANDOM_STATE)

# Combine
combined_df = pd.concat([eyd_balanced, alay_balanced, jaksel_balanced], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  Combined dataset: {len(combined_df)} total samples")
print(f"  Class distribution:\n{combined_df['label'].value_counts().to_string()}")


# ══════════════════════════════════════════════════════════════
# STEP 3: TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("\n--- Train/Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    combined_df['text'], combined_df['label'],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=combined_df['label']
)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Train distribution:\n{y_train.value_counts().to_string()}")
print(f"  Test distribution:\n{y_test.value_counts().to_string()}")


# ══════════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING (TF-IDF)
# ══════════════════════════════════════════════════════════════
print("\n--- TF-IDF Vectorization ---")

# Use both word and character n-grams
tfidf = TfidfVectorizer(
    analyzer='char_wb',      # character n-grams within word boundaries
    ngram_range=(2, 5),      # 2 to 5 char n-grams (captures slang spelling patterns)
    max_features=50000,
    min_df=2,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"  Feature dimension: {X_train_tfidf.shape[1]}")
print(f"  Train matrix: {X_train_tfidf.shape}")
print(f"  Test matrix: {X_test_tfidf.shape}")


# ══════════════════════════════════════════════════════════════
# STEP 5: TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\n--- Training Models ---")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, random_state=RANDOM_STATE
    ),
    'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
    'Linear SVM': LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': acc,
        'y_pred': y_pred,
        'model': model,
    }
    print(f"    Accuracy: {acc:.4f}")


# ══════════════════════════════════════════════════════════════
# STEP 6: EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

# Find best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_result = results[best_name]

print(f"\n{'Model':<30} {'Accuracy':>10}")
print("-" * 42)
for name, res in results.items():
    marker = " ★" if name == best_name else ""
    print(f"{name:<30} {res['accuracy']:>10.4f}{marker}")

print(f"\nBest Model: {best_name}")

# Detailed report for best model
target_names = ['alay', 'eyd', 'jaksel']
print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, best_result['y_pred'], 
                           target_names=target_names,
                           digits=4))

print(f"--- Confusion Matrix ({best_name}) ---")
cm = confusion_matrix(y_test, best_result['y_pred'], labels=target_names)
cm_df = pd.DataFrame(cm, index=target_names, 
                     columns=[f'pred_{t}' for t in target_names])
print(cm_df.to_string())


# ══════════════════════════════════════════════════════════════
# STEP 7: SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════
print("\n--- Saving Best Model ---")

model_data = {
    'model': best_result['model'],
    'vectorizer': tfidf,
    'classes': target_names,
    'best_model_name': best_name,
    'accuracy': best_result['accuracy'],
}

with open('sociolinguistics_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"  Saved: sociolinguistics_model.pkl")


# ══════════════════════════════════════════════════════════════
# STEP 8: INTERACTIVE DEMO
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEMO: Classifying Sample Texts")
print("=" * 60)

demo_texts = [
    # Alay/Slang
    "gw bgt lah, emg gk bsa tidur smpe pagi",
    "kk cantik bgt ya, sukses slalu",
    "tp gw kgn bgt sma lo, kpn ktmu lg?",
    
    # Jaksel (Indonglish)
    "Literally gue tuh lagi overthinking banget deh",
    "So basically dia tuh toxic banget, I can't even",
    "Ngl sih healing kemarin vibes nya on point banget",
    
    # EYD (Formal)
    "Pemerintah telah menetapkan kebijakan baru mengenai kesehatan masyarakat.",
    "Dengan hormat, kami sampaikan undangan untuk menghadiri acara tersebut.",
    "Pendidikan merupakan fondasi utama dalam pembangunan karakter bangsa.",
    
    # Additional EYD-style (literary)
    "Zaman menghela nafas panjang, menatap keluar jendela kantor.",
    "Mereka berdua mengenal satu sama lain dengan baik lewat rangkaian percakapan pendek.",
]

best_model = best_result['model']
for text in demo_texts:
    vec = tfidf.transform([text])
    pred = best_model.predict(vec)[0]
    print(f"\n  [{pred:>6}] {text[:80]}")


# ══════════════════════════════════════════════════════════════
# ALSO: Save Word-level TF-IDF Model for comparison
# ══════════════════════════════════════════════════════════════
print("\n\n--- Bonus: Word-level TF-IDF Model ---")

tfidf_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    sublinear_tf=True,
)

X_train_word = tfidf_word.fit_transform(X_train)
X_test_word = tfidf_word.transform(X_test)

lr_word = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
lr_word.fit(X_train_word, y_train)
y_pred_word = lr_word.predict(X_test_word)
acc_word = accuracy_score(y_test, y_pred_word)
print(f"  Word-level LR Accuracy: {acc_word:.4f}")
print(f"  (vs char n-gram best: {best_result['accuracy']:.4f})")

print(classification_report(y_test, y_pred_word,
                           target_names=target_names,
                           digits=4))


print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
