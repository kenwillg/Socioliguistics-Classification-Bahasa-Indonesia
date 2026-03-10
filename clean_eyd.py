"""
Clean the eyd_tere_liye_tentang_kamu.txt (EYD / Formal Indonesian dataset)
Source: https://archive.org/stream/tentangkamu/TENTANG%20KAMU_djvu.txt
Book: Tere Liye - Tentang Kamu
"""

import pandas as pd
import re
import os

def clean_eyd_text(raw_path, output_path):
    print("=" * 60)
    print("CLEANING: EYD Dataset (Tere Liye — Tentang Kamu)")
    print("=" * 60)

    # Read raw file
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    print(f"\nRaw file: {len(raw_lines)} lines")

    # ── Step 1: Remove "E B O O K EXCLUSIVE" lines and surrounding blanks ──
    cleaned_lines = []
    skip_next_blanks = False
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()

        # Check for E B O O K EXCLUSIVE pattern
        if re.match(r'^E\s*B\s*O\s*O\s*K\s+EXCLUSIVE\s*$', line, re.IGNORECASE):
            # Skip this line and surrounding blank lines
            # Look backwards and remove trailing blank lines we already added
            while cleaned_lines and cleaned_lines[-1].strip() == '':
                cleaned_lines.pop()
            # Skip forward past blank lines after the marker
            i += 1
            while i < len(raw_lines) and raw_lines[i].strip() == '':
                i += 1
            continue

        cleaned_lines.append(raw_lines[i])
        i += 1

    print(f"After removing E B O O K EXCLUSIVE: {len(cleaned_lines)} lines")

    # ── Step 2: Remove metadata, headers, publisher info ──
    # Remove first few lines (title, author) and last lines (publisher info)
    # Find where actual content starts (BAB 1)
    content_start = 0
    content_end = len(cleaned_lines)

    for idx, line in enumerate(cleaned_lines):
        if re.match(r'^BAB\s+1\.', line.strip()):
            content_start = idx
            break

    # Find where content ends (publisher info, etc.)
    for idx in range(len(cleaned_lines) - 1, -1, -1):
        line = cleaned_lines[idx].strip()
        if line and not re.match(r'^(www\.|Kav\.|Jakarta|Telp\.|PENERBIT|\^|Novel|Kepustakaan|PAB)', line):
            content_end = idx + 1
            break

    cleaned_lines = cleaned_lines[content_start:content_end]
    print(f"After removing metadata: {len(cleaned_lines)} lines")

    # ── Step 3: Join lines into paragraphs, then split into sentences ──
    # Join continuation lines (lines that don't start a new paragraph)
    paragraphs = []
    current_para = []

    for line in cleaned_lines:
        stripped = line.strip()

        # Skip chapter headings (BAB X. Title)
        if re.match(r'^BAB\s+\d+\.', stripped):
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            continue

        # Skip separator lines (*** or ---)
        if re.match(r'^[\*\-]{3,}$', stripped):
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            continue

        # Empty line = paragraph break
        if stripped == '':
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            continue

        # Otherwise, it's a continuation of the current paragraph
        current_para.append(stripped)

    if current_para:
        paragraphs.append(' '.join(current_para))

    print(f"Paragraphs extracted: {len(paragraphs)}")

    # ── Step 4: Split paragraphs into sentences ──
    sentences = []
    for para in paragraphs:
        # Clean up multiple spaces
        para = re.sub(r'\s+', ' ', para).strip()

        # Split on sentence-ending punctuation, keeping the punctuation
        # Handle cases like "Mr.", "Dr.", numbers like "07.30"
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', para)

        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    print(f"Sentences extracted: {len(sentences)}")

    # ── Step 5: Clean and filter sentences ──
    final_sentences = []
    for sent in sentences:
        # Remove quotes that are dialogue markers (keep the text)
        sent = sent.strip()

        # Remove sentences that are just dialogue tags or very short
        if len(sent) < 15:
            continue

        # Remove sentences that are mostly non-Indonesian (URLs, codes, etc.)
        if re.match(r'^(www\.|http|ISBN|\d{10,})', sent):
            continue

        # Remove lines that look like page numbers or codes
        if re.match(r'^[\d\s]+$', sent):
            continue

        # Remove excessive special characters
        alpha_ratio = sum(1 for c in sent if c.isalpha()) / max(len(sent), 1)
        if alpha_ratio < 0.5:
            continue

        final_sentences.append(sent)

    print(f"After filtering: {len(final_sentences)} sentences")

    # ── Step 6: Build and save DataFrame ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    eyd_df = pd.DataFrame({
        'text': final_sentences,
        'label': 'eyd'
    })

    # Remove duplicates
    eyd_df = eyd_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"After deduplication: {len(eyd_df)} sentences")

    eyd_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(eyd_df)} rows)")

    # ── Stats ──
    print("\n--- Quick Stats ---")
    print(f"Avg text length: {eyd_df['text'].str.len().mean():.1f} chars")
    print(f"Min text length: {eyd_df['text'].str.len().min()} chars")
    print(f"Max text length: {eyd_df['text'].str.len().max()} chars")
    print(f"Avg word count: {eyd_df['text'].str.split().str.len().mean():.1f} words")

    # Show samples
    print("\n--- Sample Sentences ---")
    for i, row in eyd_df.sample(min(10, len(eyd_df)), random_state=42).iterrows():
        print(f"  [{i}] {row['text'][:120]}...")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    return eyd_df


def main():
    clean_eyd_text(
        raw_path='source-data/eyd_tere_liye_tentang_kamu.txt',
        output_path='cleaned-data/eyd_cleaned.csv'
    )


if __name__ == '__main__':
    main()
