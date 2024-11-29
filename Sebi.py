# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import hashlib
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')

# Fungsi untuk membersihkan teks (Cleansing)
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.S+')
    return url.sub(r'', text)

def remove_html(text):
    html = re.compile(r'<.#?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Preprocessing teks
def preprocess_text(text):
    if isinstance(text, str):
        text = remove_url(text)
        text = remove_html(text)
        text = remove_emoji(text)
        text = remove_numbers(text)
        text = remove_symbols(text)
        text = text.lower()
        tokens = text.split()
        stop_words = set(stopwords.words('indonesian'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    else:
        return ""

# Fungsi untuk TF-IDF Vectorization
def compute_tfidf(data, column='stopword_removal'):
    global vectorizer, tfidf_matrix  # Menyatakan variabel ini sebagai global
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data[column])
    return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Fungsi pencarian berita serupa
def search_similar(query, top_n=5):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    similarities = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['judul', 'kategori', 'cleansed_text']]

# Layout Aplikasi Streamlit
st.title("Aplikasi SEBI - Pengambilan dan Analisis Berita")

# Upload file CSV
uploaded_file = st.file_uploader("Upload CSV Hasil Crawling", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Berita:")
    st.dataframe(df.head(10))

    # Preprocessing
    st.subheader("Preprocessing Data")
    df['cleansed_text'] = df['isi'].apply(preprocess_text)
    st.write("Data setelah Preprocessing:")
    st.dataframe(df[['judul', 'cleansed_text']].head(10))

    # TF-IDF Vectorization
    st.subheader("Hasil TF-IDF Vectorization")
    tfidf_df = compute_tfidf(df, column='cleansed_text')
    st.write("TF-IDF DataFrame:")
    st.dataframe(tfidf_df.head(10))

    # Visualisasi TF-IDF
    st.subheader("Visualisasi TF-IDF")
    top_n = st.slider("Pilih jumlah kata yang akan ditampilkan", 10, 50, 20)
    top_words = tfidf_df.sum().sort_values(ascending=False).head(top_n)
    st.bar_chart(top_words)

    # Pencarian berita serupa
    st.subheader("Cari Berita Serupa")
    query_input = st.text_area("Masukkan kata kunci pencarian:")
    if query_input:
        results = search_similar(query_input, top_n=5)
        st.write("Berita Serupa:")
        st.dataframe(results)
