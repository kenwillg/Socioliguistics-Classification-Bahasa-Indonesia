"""
Indonesian Sociolinguistics Text Classifier
3-class classification: Bahasa Baku (formal), Alay/Slang, Jaksel (Indonglish)

Strategy:
  - Alay samples: context sentences from colloquial lexicon (contain slang)
  - Jaksel samples: code-switched tweets (Indonesian-English mix)
  - Baku samples: formal Indonesian words/phrases from the slang dictionary's
    'formal' column, supplemented by reconstructing clean sentences
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

# ── Load Data ───────────────────────────────────────────────
print("=" * 60)
print("INDONESIAN SOCIOLINGUISTICS CLASSIFIER")
print("=" * 60)

# Load cleaned datasets
alay_df = pd.read_csv('alay_cleaned.csv')
jaksel_df = pd.read_csv('jaksel_cleaned.csv')
slang_dict = pd.read_csv('slang_dictionary.csv')

print(f"\nLoaded:")
print(f"  Alay samples:  {len(alay_df)}")
print(f"  Jaksel samples: {len(jaksel_df)}")
print(f"  Slang dictionary: {len(slang_dict)} entries")


# ══════════════════════════════════════════════════════════════
# STEP 1: CREATE BAHASA BAKU SAMPLES
# ══════════════════════════════════════════════════════════════
print("\n--- Creating Bahasa Baku (formal) samples ---")

# Strategy: take the alay context sentences and replace all slang words 
# with their formal equivalents to create "baku" versions of the same texts.
# This gives us paired data: alay original → baku corrected

# Build slang→formal lookup (take most common formal per slang)
slang_to_formal = {}
for _, row in slang_dict.iterrows():
    slang_word = str(row['slang']).lower().strip()
    formal_word = str(row['formal']).lower().strip()
    if slang_word and formal_word and slang_word != 'nan' and formal_word != 'nan':
        slang_to_formal[slang_word] = formal_word

print(f"  Slang→Formal lookup: {len(slang_to_formal)} mappings")

def convert_to_baku(text):
    """Replace slang words with formal equivalents."""
    if not isinstance(text, str):
        return ""
    words = text.lower().split()
    result = []
    for w in words:
        # Strip punctuation for lookup
        clean_w = re.sub(r'[^\w]', '', w)
        if clean_w in slang_to_formal:
            result.append(slang_to_formal[clean_w])
        else:
            result.append(w.lower())
    return ' '.join(result)

# Convert alay texts to baku
baku_texts = alay_df['text'].apply(convert_to_baku)

# Remove baku texts that are identical to alay (no slang was replaced)
# and remove very short texts
baku_df = pd.DataFrame({'text': baku_texts, 'label': 'baku'})
baku_df = baku_df[baku_df['text'].str.len() >= 5]

# Also add some purely formal sentences from common Indonesian formal phrases
formal_phrases = [
    "Dengan hormat, saya bermaksud menyampaikan surat ini.",
    "Berdasarkan hasil rapat yang telah dilaksanakan.",
    "Sehubungan dengan hal tersebut di atas.",
    "Demikian surat ini dibuat untuk dapat dipergunakan sebagaimana mestinya.",
    "Atas perhatian dan kerjasamanya, kami ucapkan terima kasih.",
    "Pada kesempatan ini, kami ingin menyampaikan beberapa hal penting.",
    "Sesuai dengan peraturan yang berlaku, kami mohon perhatian.",
    "Dengan ini kami sampaikan laporan kegiatan bulan ini.",
    "Berkenaan dengan undangan yang telah disampaikan.",
    "Kami mengharapkan kehadiran Bapak atau Ibu pada acara tersebut.",
    "Pemerintah telah menetapkan kebijakan baru mengenai pendidikan.",
    "Indonesia merupakan negara kepulauan yang memiliki kekayaan alam.",
    "Pembangunan infrastruktur terus dilakukan untuk kemajuan bangsa.",
    "Pendidikan merupakan kunci utama dalam membangun sumber daya manusia.",
    "Kegiatan ini bertujuan untuk meningkatkan kesadaran masyarakat.",
    "Setiap warga negara memiliki hak dan kewajiban yang sama.",
    "Semoga kegiatan ini dapat berjalan dengan lancar dan sukses.",
    "Masyarakat diimbau untuk menjaga kebersihan lingkungan sekitar.",
    "Pelaksanaan program ini memerlukan kerjasama dari berbagai pihak.",
    "Kami berharap dapat memberikan pelayanan yang terbaik kepada masyarakat.",
    "Ilmu pengetahuan dan teknologi berkembang dengan sangat pesat.",
    "Kesehatan merupakan hal yang sangat penting dalam kehidupan.",
    "Pertumbuhan ekonomi Indonesia menunjukkan tren positif tahun ini.",
    "Keanekaragaman budaya Indonesia menjadi kekayaan yang harus dijaga.",
    "Program kerja yang telah disusun akan segera dilaksanakan.",
    "Seluruh peserta diharapkan dapat mengikuti kegiatan dengan tertib.",
    "Laporan keuangan telah disusun sesuai dengan standar akuntansi.",
    "Hasil penelitian menunjukkan adanya peningkatan yang signifikan.",
    "Pihak berwenang telah mengambil tindakan yang diperlukan.",
    "Kerja sama antara kedua belah pihak diharapkan dapat terjalin.",
]
formal_addition = pd.DataFrame({'text': formal_phrases, 'label': 'baku'})
baku_df = pd.concat([baku_df, formal_addition], ignore_index=True)

# Deduplicate
baku_df = baku_df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"  Baku samples created: {len(baku_df)}")


# ══════════════════════════════════════════════════════════════
# STEP 2: BALANCE AND COMBINE DATASETS
# ══════════════════════════════════════════════════════════════
print("\n--- Combining datasets ---")

# Check class sizes
print(f"  Alay: {len(alay_df)}")
print(f"  Jaksel: {len(jaksel_df)}")
print(f"  Baku: {len(baku_df)}")

# Balance by downsampling to the smallest class size
min_size = min(len(alay_df), len(jaksel_df), len(baku_df))
print(f"  Balancing to: {min_size} samples per class")

alay_balanced = alay_df.sample(n=min_size, random_state=RANDOM_STATE)
jaksel_balanced = jaksel_df.sample(n=min_size, random_state=RANDOM_STATE)
baku_balanced = baku_df.sample(n=min_size, random_state=RANDOM_STATE)

# Combine
combined_df = pd.concat([alay_balanced, jaksel_balanced, baku_balanced], ignore_index=True)
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
print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, best_result['y_pred'], 
                           target_names=['alay', 'baku', 'jaksel'],
                           digits=4))

print(f"--- Confusion Matrix ({best_name}) ---")
cm = confusion_matrix(y_test, best_result['y_pred'], labels=['alay', 'baku', 'jaksel'])
cm_df = pd.DataFrame(cm, index=['alay', 'baku', 'jaksel'], columns=['pred_alay', 'pred_baku', 'pred_jaksel'])
print(cm_df.to_string())


# ══════════════════════════════════════════════════════════════
# STEP 7: SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════
print("\n--- Saving Best Model ---")

model_data = {
    'model': best_result['model'],
    'vectorizer': tfidf,
    'classes': ['alay', 'baku', 'jaksel'],
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
    
    # Baku (Formal)
    "Pemerintah telah menetapkan kebijakan baru mengenai kesehatan masyarakat.",
    "Dengan hormat, kami sampaikan undangan untuk menghadiri acara tersebut.",
    "Pendidikan merupakan fondasi utama dalam pembangunan karakter bangsa.",
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
                           target_names=['alay', 'baku', 'jaksel'],
                           digits=4))


print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
