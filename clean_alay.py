"""
Clean the colloquial-indonesian-lexicon.csv (Alay/Slang dataset)
Source: https://github.com/nasalsabila/kamus-alay
"""

import pandas as pd
import re
import os

SOURCE_DIR = 'source-data'
OUTPUT_DIR = 'cleaned-data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    # Remove @mentions
    text = re.sub(r'@\S+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove BBM pins, phone numbers
    text = re.sub(r'[A-Z0-9]{8,}', '', text)
    # Remove emojis and special unicode
    text = re.sub(r'[^\w\s.,!?\'\"-]', '', text, flags=re.UNICODE)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=" * 60)
    print("CLEANING: Alay/Slang Dataset")
    print("=" * 60)

    # Load raw data
    df = pd.read_csv(os.path.join(SOURCE_DIR, 'colloquial-indonesian-lexicon.csv'))
    print(f"\nRaw dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample rows:")
    print(df.head(10).to_string())

    # --- Clean the context column (full sentences with slang) ---
    print("\n--- Preparing ALAY text samples from 'context' column ---")
    
    # Extract unique context sentences
    alay_contexts = df['context'].dropna().drop_duplicates().reset_index(drop=True)
    print(f"Unique context sentences: {len(alay_contexts)}")

    # Clean each sentence
    alay_texts = alay_contexts.apply(clean_text)
    
    # Remove empty strings and very short texts (< 5 chars)
    alay_texts = alay_texts[alay_texts.str.len() >= 5]
    print(f"After cleaning (>= 5 chars): {len(alay_texts)}")

    # Build alay cleaned dataframe
    alay_df = pd.DataFrame({
        'text': alay_texts.values,
        'label': 'alay'
    })

    # --- Also extract formal equivalents for Bahasa Baku ---
    print("\n--- Preparing BAKU text samples from 'formal' column ---")
    
    # Get unique formal words/phrases
    baku_words = df['formal'].dropna().drop_duplicates().reset_index(drop=True)
    print(f"Unique formal words: {len(baku_words)}")
    
    # For baku, we need sentences, not just words.
    # We'll create simple sentences from the formal words by 
    # reconstructing the context with formal replacements
    # But for now, let's just save the slang-formal mapping separately
    
    # Save the slang-to-formal dictionary
    slang_dict = df[['slang', 'formal']].dropna().drop_duplicates()
    slang_dict.to_csv(os.path.join(OUTPUT_DIR, 'slang_dictionary.csv'), index=False)
    print(f"Saved slang dictionary: {len(slang_dict)} entries -> {OUTPUT_DIR}/slang_dictionary.csv")

    # Save cleaned alay dataset
    alay_df.to_csv(os.path.join(OUTPUT_DIR, 'alay_cleaned.csv'), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/alay_cleaned.csv ({len(alay_df)} rows)")
    
    # --- Stats ---
    print("\n--- Quick Stats ---")
    print(f"Avg text length: {alay_df['text'].str.len().mean():.1f} chars")
    print(f"Min text length: {alay_df['text'].str.len().min()} chars")
    print(f"Max text length: {alay_df['text'].str.len().max()} chars")
    print(f"Avg word count: {alay_df['text'].str.split().str.len().mean():.1f} words")
    
    # --- Also export slang categories for EDA ---
    categories = df[['slang', 'formal', 'category1', 'category2', 'category3']].copy()
    categories.to_csv(os.path.join(OUTPUT_DIR, 'alay_categories.csv'), index=False)
    print(f"\nSaved category metadata: {OUTPUT_DIR}/alay_categories.csv")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
