import pandas as pd

new_sentences = [
    "Saya ingin pergi ke supermarket untuk berbelanja sayur dan buah.",
    "Hari ini cuaca sangat cerah sehingga cocok untuk berolahraga.",
    "Buku ini memberikan wawasan baru tentang pentingnya menjaga lingkungan.",
    "Mahasiswa diharapkan mampu menyusun karya ilmiah dengan baik.",
    "Pemerintah sedang berupaya meningkatkan kesejahteraan masyarakat secara merata.",
    "Kami memohon maaf atas ketidaknyamanan yang terjadi selama perbaikan jalan.",
    "Mohon periksa kembali jadwal keberangkatan kereta api Anda.",
    "Sistem pendaftaran online akan ditutup pada akhir bulan ini.",
    "Penelitian menunjukkan bahwa olahraga teratur dapat meningkatkan fungsi kognitif.",
    "Terima kasih atas perhatian dan kerja sama yang diberikan."
]

new_df = pd.DataFrame({'text': new_sentences, 'label': ['eyd'] * len(new_sentences)})

df = pd.read_csv('cleaned-data/eyd_cleaned.csv', encoding='utf-8')
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv('cleaned-data/eyd_cleaned.csv', index=False, encoding='utf-8')
print("Successfully appended 10 new formal sentences to eyd_cleaned.csv")
