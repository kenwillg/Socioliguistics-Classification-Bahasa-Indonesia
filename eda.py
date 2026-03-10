"""
Exploratory Data Analysis for Indonesian Sociolinguistics Datasets
Analyzes: eyd_cleaned.csv, alay_cleaned.csv, jaksel_cleaned.csv, alay_categories.csv
"""

import pandas as pd
import numpy as np
import re
import os
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────
DATA_DIR = 'cleaned-data'
OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
COLORS = {
    'eyd': '#45B7D1',
    'alay': '#FF6B6B',
    'jaksel': '#4ECDC4',
}

# ── Load Data ───────────────────────────────────────────────────
print("Loading datasets...")
eyd_df = pd.read_csv(os.path.join(DATA_DIR, 'eyd_cleaned.csv'))
alay_df = pd.read_csv(os.path.join(DATA_DIR, 'alay_cleaned.csv'))
jaksel_df = pd.read_csv(os.path.join(DATA_DIR, 'jaksel_cleaned.csv'))
categories_df = pd.read_csv(os.path.join(DATA_DIR, 'alay_categories.csv'))

print(f"EYD dataset: {len(eyd_df)} rows")
print(f"Alay dataset: {len(alay_df)} rows")
print(f"Jaksel dataset: {len(jaksel_df)} rows")
print(f"Categories metadata: {len(categories_df)} rows")


# ── Helper Functions ────────────────────────────────────────────
def count_english_words(text):
    """Rough heuristic: count words that look English."""
    english_markers = {
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'need',
        'a', 'an', 'and', 'but', 'or', 'not', 'no', 'nor',
        'for', 'of', 'to', 'in', 'on', 'at', 'by', 'with',
        'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between',
        'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your',
        'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
        'this', 'that', 'these', 'those', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'if', 'then', 'so', 'than', 'too', 'very', 'just',
        'because', 'as', 'until', 'while', 'although', 'though',
        'even', 'also', 'still', 'already', 'ever', 'never',
        'always', 'sometimes', 'often', 'usually', 'here', 'there',
        'now', 'then', 'today', 'tomorrow', 'yesterday',
        'love', 'like', 'want', 'know', 'think', 'feel',
        'make', 'go', 'come', 'see', 'get', 'give', 'take',
        'good', 'bad', 'happy', 'sad', 'new', 'old', 'big', 'small',
        'really', 'actually', 'literally', 'honestly', 'basically',
        'someone', 'something', 'anything', 'everything', 'nothing',
        'people', 'person', 'time', 'way', 'day', 'thing', 'life',
        'much', 'many', 'more', 'most', 'some', 'any', 'few',
        'only', 'own', 'same', 'other', 'another',
        'well', 'back', 'right', 'over', 'out', 'off',
        'healing', 'overthinking', 'literally', 'bestie', 'crush',
        'vibes', 'mood', 'support', 'system', 'toxic', 'self',
    }
    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0
    eng_count = sum(1 for w in words if w in english_markers)
    return eng_count / len(words)

def get_word_freq(texts, top_n=30):
    """Get top N word frequencies."""
    all_words = ' '.join(texts.astype(str)).lower().split()
    # Remove very short words
    all_words = [w for w in all_words if len(w) > 2]
    return Counter(all_words).most_common(top_n)

def count_special_chars(text):
    """Count ratio of digits + special chars."""
    text = str(text)
    if len(text) == 0:
        return 0.0
    special = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return special / len(text)


# ══════════════════════════════════════════════════════════════
# 1. BASIC STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. BASIC STATISTICS")
print("=" * 60)

all_datasets = [
    ('EYD (Formal)', eyd_df),
    ('Alay/Slang', alay_df),
    ('Jaksel/Indonglish', jaksel_df),
]

for name, df in all_datasets:
    df['text_len'] = df['text'].astype(str).str.len()
    df['word_count'] = df['text'].astype(str).str.split().str.len()
    df['special_ratio'] = df['text'].apply(count_special_chars)
    
    print(f"\n--- {name} ---")
    print(f"  Total rows: {len(df)}")
    print(f"  Avg text length: {df['text_len'].mean():.1f} chars")
    print(f"  Median text length: {df['text_len'].median():.1f} chars")
    print(f"  Avg word count: {df['word_count'].mean():.1f} words")
    print(f"  Median word count: {df['word_count'].median():.1f} words")
    print(f"  Avg special char ratio: {df['special_ratio'].mean():.3f}")


# ══════════════════════════════════════════════════════════════
# 2. TEXT LENGTH DISTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n2. Generating text length distribution plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, df, key) in zip(axes, [
    ('EYD (Formal)', eyd_df, 'eyd'),
    ('Alay/Slang', alay_df, 'alay'),
    ('Jaksel/Indonglish', jaksel_df, 'jaksel'),
]):
    ax.hist(df['text_len'], bins=50, color=COLORS[key], alpha=0.85, edgecolor='white')
    ax.set_title(f'{name} — Text Length Distribution')
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Frequency')
    ax.axvline(df['text_len'].median(), color='black', linestyle='--',
               label=f"Median: {df['text_len'].median():.0f}")
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 01_text_length_distribution.png")


# ══════════════════════════════════════════════════════════════
# 3. WORD COUNT DISTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n3. Generating word count distribution plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, df, key) in zip(axes, [
    ('EYD (Formal)', eyd_df, 'eyd'),
    ('Alay/Slang', alay_df, 'alay'),
    ('Jaksel/Indonglish', jaksel_df, 'jaksel'),
]):
    ax.hist(df['word_count'], bins=40, color=COLORS[key], alpha=0.85, edgecolor='white')
    ax.set_title(f'{name} — Word Count Distribution')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_word_count_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 02_word_count_distribution.png")


# ══════════════════════════════════════════════════════════════
# 4. TOP WORD FREQUENCIES
# ══════════════════════════════════════════════════════════════
print("\n4. Generating word frequency plots...")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for ax, (name, df, key) in zip(axes, [
    ('EYD (Formal)', eyd_df, 'eyd'),
    ('Alay/Slang', alay_df, 'alay'),
    ('Jaksel/Indonglish', jaksel_df, 'jaksel'),
]):
    freq = get_word_freq(df['text'], top_n=25)
    words, counts = zip(*freq)
    y_pos = range(len(words))
    ax.barh(y_pos, counts, color=COLORS[key], alpha=0.85, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f'{name} — Top 25 Words')
    ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_top_word_frequencies.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 03_top_word_frequencies.png")


# ══════════════════════════════════════════════════════════════
# 5. SLANG CATEGORY DISTRIBUTION (Alay only)
# ══════════════════════════════════════════════════════════════
print("\n5. Generating slang category distribution...")

cat1_counts = categories_df['category1'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(12, 6))
cat1_counts.plot(kind='barh', color=COLORS['alay'], alpha=0.85, edgecolor='white', ax=ax)
ax.set_title('Alay Slang — Category Distribution (Primary)')
ax.set_xlabel('Count')
ax.set_ylabel('Category')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_slang_category_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 04_slang_category_distribution.png")

# Print category stats
print("\n  Slang Categories (top 10):")
for cat, count in cat1_counts.head(10).items():
    print(f"    {cat}: {count}")


# ══════════════════════════════════════════════════════════════
# 6. ENGLISH MIXING RATIO (Jaksel)
# ══════════════════════════════════════════════════════════════
print("\n6. Analyzing English word mixing in Jaksel dataset...")

jaksel_df['eng_ratio'] = jaksel_df['text'].apply(count_english_words)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(jaksel_df['eng_ratio'], bins=50, color=COLORS['jaksel'], alpha=0.85, edgecolor='white')
ax.set_title('Jaksel — English Word Ratio Distribution')
ax.set_xlabel('English Word Ratio')
ax.set_ylabel('Frequency')
ax.axvline(jaksel_df['eng_ratio'].median(), color='black', linestyle='--',
           label=f"Median: {jaksel_df['eng_ratio'].median():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_jaksel_english_ratio.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Avg English word ratio: {jaksel_df['eng_ratio'].mean():.3f}")
print(f"  Median English word ratio: {jaksel_df['eng_ratio'].median():.3f}")
print(f"  Tweets with >50% English: {(jaksel_df['eng_ratio'] > 0.5).sum()}")
print(f"  Tweets with >25% English: {(jaksel_df['eng_ratio'] > 0.25).sum()}")
print("  → Saved 05_jaksel_english_ratio.png")


# ══════════════════════════════════════════════════════════════
# 7. COMPARISON BOXPLOT (all 3 classes)
# ══════════════════════════════════════════════════════════════
print("\n7. Generating comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Text length comparison
data_len = [eyd_df['text_len'], alay_df['text_len'], jaksel_df['text_len']]
bp1 = axes[0].boxplot(data_len, labels=['EYD', 'Alay/Slang', 'Jaksel'], 
                       patch_artist=True, widths=0.5)
bp1['boxes'][0].set_facecolor(COLORS['eyd'])
bp1['boxes'][1].set_facecolor(COLORS['alay'])
bp1['boxes'][2].set_facecolor(COLORS['jaksel'])
axes[0].set_title('Text Length Comparison (3 Classes)')
axes[0].set_ylabel('Character Count')

# Word count comparison
data_wc = [eyd_df['word_count'], alay_df['word_count'], jaksel_df['word_count']]
bp2 = axes[1].boxplot(data_wc, labels=['EYD', 'Alay/Slang', 'Jaksel'],
                       patch_artist=True, widths=0.5)
bp2['boxes'][0].set_facecolor(COLORS['eyd'])
bp2['boxes'][1].set_facecolor(COLORS['alay'])
bp2['boxes'][2].set_facecolor(COLORS['jaksel'])
axes[1].set_title('Word Count Comparison (3 Classes)')
axes[1].set_ylabel('Word Count')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_comparison_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 06_comparison_boxplot.png")


# ══════════════════════════════════════════════════════════════
# 8. DATASET SIZE OVERVIEW
# ══════════════════════════════════════════════════════════════
print("\n8. Generating dataset overview...")

fig, ax = plt.subplots(figsize=(8, 5))
datasets = ['EYD (Formal)', 'Alay/Slang', 'Jaksel/Indonglish']
sizes = [len(eyd_df), len(alay_df), len(jaksel_df)]
colors = [COLORS['eyd'], COLORS['alay'], COLORS['jaksel']]
bars = ax.bar(datasets, sizes, color=colors, alpha=0.85, edgecolor='white', width=0.5)
ax.set_title('Dataset Sizes (3 Classes)')
ax.set_ylabel('Number of Samples')

# Add count labels on bars
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
            f'{size:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_dataset_sizes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 07_dataset_sizes.png")


# ══════════════════════════════════════════════════════════════
# 9. EYD SENTENCE LENGTH DISTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n9. EYD-specific analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sentence length distribution for EYD
axes[0].hist(eyd_df['text_len'], bins=60, color=COLORS['eyd'], alpha=0.85, edgecolor='white')
axes[0].set_title('EYD — Sentence Length Distribution')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].axvline(eyd_df['text_len'].median(), color='black', linestyle='--',
               label=f"Median: {eyd_df['text_len'].median():.0f}")
axes[0].legend()

# English ratio comparison across all 3 classes
eyd_df['eng_ratio'] = eyd_df['text'].apply(count_english_words)
alay_df['eng_ratio'] = alay_df['text'].apply(count_english_words)

eng_data = [eyd_df['eng_ratio'], alay_df['eng_ratio'], jaksel_df['eng_ratio']]
bp3 = axes[1].boxplot(eng_data, labels=['EYD', 'Alay', 'Jaksel'],
                       patch_artist=True, widths=0.5)
bp3['boxes'][0].set_facecolor(COLORS['eyd'])
bp3['boxes'][1].set_facecolor(COLORS['alay'])
bp3['boxes'][2].set_facecolor(COLORS['jaksel'])
axes[1].set_title('English Word Ratio Comparison')
axes[1].set_ylabel('English Word Ratio')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_eyd_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved 08_eyd_analysis.png")


# ══════════════════════════════════════════════════════════════
# 10. SAMPLE TEXTS
# ══════════════════════════════════════════════════════════════
print("\n10. Sample texts from each dataset:")

print("\n--- EYD (Formal) Samples ---")
for i, row in eyd_df.sample(5, random_state=42).iterrows():
    print(f"  [{i}] {row['text'][:120]}...")

print("\n--- Alay/Slang Samples ---")
for i, row in alay_df.sample(5, random_state=42).iterrows():
    print(f"  [{i}] {row['text'][:120]}...")

print("\n--- Jaksel/Indonglish Samples ---")
for i, row in jaksel_df.sample(5, random_state=42).iterrows():
    print(f"  [{i}] {row['text'][:120]}...")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"""
Datasets:
  EYD (Formal):      {len(eyd_df):,} samples (avg {eyd_df['word_count'].mean():.0f} words/sample)
  Alay/Slang:        {len(alay_df):,} samples (avg {alay_df['word_count'].mean():.0f} words/sample)
  Jaksel/Indonglish: {len(jaksel_df):,} samples (avg {jaksel_df['word_count'].mean():.0f} words/sample)

Key Findings:
  - EYD text is formal literary prose with longer sentences
  - Alay text is typically shorter social media comments with slang
  - Jaksel text contains English-Indonesian code-switching
  - Avg English word ratio — EYD: {eyd_df['eng_ratio'].mean():.1%}, Alay: {alay_df['eng_ratio'].mean():.1%}, Jaksel: {jaksel_df['eng_ratio'].mean():.1%}
  - Most common slang category: {cat1_counts.index[0]} ({cat1_counts.iloc[0]} occurrences)
  - Total slang categories: {categories_df['category1'].nunique()}

Output plots saved to: {OUTPUT_DIR}/
""")

print("EDA complete!")
