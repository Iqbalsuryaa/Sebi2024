import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import hashlib
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fungsi untuk membersihkan konten dari elemen-elemen yang tidak diinginkan
def clean_content(content_element):
    if content_element:
        for daftar_isi in ["collapsible"]:
            unwanted = content_element.find("div", id=daftar_isi)
            if unwanted:
                unwanted.decompose()

        for tag_class in ["aevp", "detail__body-tag mgt-16"]:
            unwanted = content_element.find_all("div", class_=tag_class)
            for el in unwanted:
                el.decompose()

        link_sisip = content_element.find_all("table", class_="linksisip")
        for table in link_sisip:
            table.decompose()

        unwanted_paragraphs = content_element.find_all(["p", "span"], class_="para_caption")
        for para in unwanted_paragraphs:
            para.decompose()

        return content_element.get_text(separator=' ', strip=True).strip()

    return "Content Not Found"

# Fungsi untuk melakukan crawling data pada situs web Detik.com
def get_data(url, kategori, min_articles_per_category):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("article", class_="list-content__item")

    for article in articles:
        if len([k for k in kategori_list if k == kategori]) >= min_articles_per_category:
            return  # Menghentikan proses jika jumlah artikel sudah mencapai minimum yang diinginkan

        link = article.find("a")["href"]
        article_response = requests.get(link)
        article_response.raise_for_status()

        article_soup = BeautifulSoup(article_response.content, "html.parser")
        title_element = article_soup.find("h1", class_="detail__title")
        title = title_element.text.strip() if title_element else "Title Not Found"
        date_element = article_soup.find("div", class_="detail__date")
        date = date_element.text.strip() if date_element else "Date Not Found"
        content_element = article_soup.find("div", class_="detail__body-text")
        content = clean_content(content_element)

        judul.append(title)
        tanggal.append(date)
        isi.append(content)
        kategori_list.append(kategori)

        if len(judul) <= 100:
            st.write(title)
        time.sleep(1)

# Preprocessing Data
def remove_url(ulasan):
    url = re.compile(r'https?://\S+|www\.S+')
    return url.sub(r'', ulasan)

def remove_html(ulasan):
    html = re.compile(r'<.#?>')
    return html.sub(r'', ulasan)

def remove_emoji(ulasan):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', ulasan)

def remove_numbers(ulasan):
    return re.sub(r'\d+', '', ulasan)

def remove_symbols(ulasan):
    return re.sub(r'[^a-zA-Z0-9\s]', '', ulasan)

def case_folding(text):
    return text.lower() if isinstance(text, str) else text

def tokenize(text):
    return text.split()

def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in text if word not in stop_words]

# Hash function for LSH
def hash_function(text, num_buckets=100):
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % num_buckets

# Set up Streamlit app
st.title('Aplikasi Crawling dan Analisis Berita')
st.sidebar.header('Pengaturan')

# Pilih kategori untuk crawling
kategori = st.sidebar.selectbox('Pilih kategori berita', ["Kesehatan", "Pariwisata", "Politik", "Olaraga"])
jumlah_artikel = st.sidebar.slider('Jumlah artikel yang diambil', min_value=10, max_value=100, value=20)

if st.sidebar.button('Mulai Crawling'):
    base_url = {
        "Kesehatan": "https://health.detik.com/berita-detikhealth/indeks",
        "Pariwisata": "https://travel.detik.com/travel-news/indeks",
        "Politik": "https://news.detik.com/indeks",
        "Olaraga": "https://sport.detik.com/indeks"
    }

    # Inisialisasi list untuk menyimpan data
    judul, tanggal, isi, kategori_list = [], [], [], []
    
    # Crawling data
    get_data(base_url[kategori], kategori, jumlah_artikel)

    # Simpan data ke dalam DataFrame
    df = pd.DataFrame({"judul": judul, "tanggal": tanggal, "isi": isi, "kategori": kategori_list})
    st.write('Data Crawling berhasil dikumpulkan!')
    st.dataframe(df)

    # Simpan DataFrame ke file CSV
    df.to_csv('berita_crawled.csv', index=False)

# Preprocessing data
st.header('Preprocessing Data Berita')
if st.button('Mulai Preprocessing'):
    df = pd.read_csv('berita_crawled.csv')
    df['cleansing'] = df['isi'].apply(lambda x: remove_url(x))
    df['cleansing'] = df['cleansing'].apply(lambda x: remove_html(x))
    df['cleansing'] = df['cleansing'].apply(lambda x: remove_emoji(x))
    df['cleansing'] = df['cleansing'].apply(lambda x: remove_numbers(x))
    df['cleansing'] = df['cleansing'].apply(lambda x: remove_symbols(x))
    df['case_folding'] = df['cleansing'].apply(case_folding)
    df['tokenize'] = df['case_folding'].apply(tokenize)
    df['stopword_removal'] = df['tokenize'].apply(lambda x: ' '.join(remove_stopwords(x)))

    st.write('Preprocessing selesai!')
    st.dataframe(df)

    # Simpan data setelah preprocessing
    df.to_csv('berita_preprocessed.csv', index=False)

# TF-IDF Vectorization
st.header('TF-IDF Vectorization')
if st.button('Mulai TF-IDF'):
    df = pd.read_csv('berita_preprocessed.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['stopword_removal'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    st.write('Hasil TF-IDF Vectorization!')
    st.dataframe(tfidf_df)

    # Simpan hasil TF-IDF
    tfidf_df.to_csv('hasil_tfidf_berita.csv', index=False)

# Pencarian berita mirip (search)
st.header('Pencarian Berita Mirip')
query = st.text_input('Masukkan query pencarian')
if st.button('Cari Berita Mirip'):
    if query:
        processed_query = preprocess_text(query)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['stopword_removal'])
        query_vector = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        results = [(df.iloc[idx]['judul'], similarities[idx]) for idx in top_indices]

        st.write('Hasil pencarian berita mirip:')
        for title, similarity in results:
            st.write(f'Judul: {title} | Similarity Score: {similarity:.2f}')
