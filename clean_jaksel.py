"""
Clean the dataset.csv (Jaksel/Indonglish dataset)
Contains tweets with Indonesian-English code-switching
"""

import pandas as pd
import re

def clean_text(text):
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    # Remove @mentions
    text = re.sub(r'@\S+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+|t\.co/\S+', '', text)
    # Remove hashtags symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove emojis and special unicode
    text = re.sub(r'[^\w\s.,!?\'\"-]', '', text, flags=re.UNICODE)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=" * 60)
    print("CLEANING: Jaksel/Indonglish Dataset")
    print("=" * 60)

    # Load raw data
    df = pd.read_csv('dataset.csv')
    print(f"\nRaw dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSentiment label distribution:")
    print(df['label'].value_counts())
    print(f"\nSample rows:")
    print(df[['tweet', 'label']].head(10).to_string())

    # --- Clean the tweet column ---
    print("\n--- Preparing JAKSEL text samples from 'tweet' column ---")
    
    # Use the 'tweet' column - original code-switched text
    jaksel_df = df[['tweet']].copy()
    jaksel_df = jaksel_df.rename(columns={'tweet': 'text'})
    
    # Drop nulls and duplicates
    jaksel_df = jaksel_df.dropna(subset=['text'])
    jaksel_df = jaksel_df.drop_duplicates(subset=['text'])
    print(f"Unique tweets: {len(jaksel_df)}")

    # Clean text
    jaksel_df['text'] = jaksel_df['text'].apply(clean_text)
    
    # Remove empty and very short texts
    jaksel_df = jaksel_df[jaksel_df['text'].str.len() >= 5]
    print(f"After cleaning (>= 5 chars): {len(jaksel_df)}")

    # Add label
    jaksel_df['label'] = 'jaksel'
    jaksel_df = jaksel_df.reset_index(drop=True)

    # Save cleaned dataset
    jaksel_df.to_csv('jaksel_cleaned.csv', index=False)
    print(f"\nSaved: jaksel_cleaned.csv ({len(jaksel_df)} rows)")
    
    # --- Stats ---
    print("\n--- Quick Stats ---")
    print(f"Avg text length: {jaksel_df['text'].str.len().mean():.1f} chars")
    print(f"Min text length: {jaksel_df['text'].str.len().min()} chars")
    print(f"Max text length: {jaksel_df['text'].str.len().max()} chars")
    print(f"Avg word count: {jaksel_df['text'].str.split().str.len().mean():.1f} words")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
