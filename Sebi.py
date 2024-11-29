import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
import nltk

# Load pre-trained models and data
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the preprocessed data
@st.cache
def load_data():
    return pd.read_csv('Hasil_Prepros_Berita/Hasil_Prepros_Berita.csv')

df = load_data()

# Display the title of the app
st.title('Aplikasi Klasifikasi Berita dengan Streamlit')

# Sidebar for navigation
st.sidebar.header('Menu')
options = ['Lihat Data', 'Klasifikasi Berita', 'Preprocessing Data']
choice = st.sidebar.selectbox('Pilih Menu:', options)

# Show raw data in 'Lihat Data' section
if choice == 'Lihat Data':
    st.subheader('Data Berita')
    st.dataframe(df.head(10))

# Preprocessing function
def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Remove URLs, numbers, and special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Klasifikasi berita dengan model logistic regression
if choice == 'Klasifikasi Berita':
    st.subheader('Klasifikasi Berita')
    input_text = st.text_area('Masukkan Berita untuk Diklasifikasikan:', '')
    
    if st.button('Klasifikasi'):
        if input_text:
            processed_text = preprocess_text(input_text)
            # Vectorize the input text
            input_vector = vectorizer.transform([processed_text])
            # Predict the category
            prediction = model.predict(input_vector)
            st.write(f'Prediksi Kategori Berita: {prediction[0]}')
        else:
            st.error('Harap masukkan teks berita.')

# Processing data for 'Preprocessing Data' section
if choice == 'Preprocessing Data':
    st.subheader('Preprocessing Data Berita')
    st.write('Preprocessing dilakukan pada data crawling untuk membersihkan teks sebelum klasifikasi.')
    if st.button('Tampilkan Data Preprocessed'):
        df_preprocessed = pd.read_csv('Hasil_Prepros_Berita/Hasil_Prepros_Berita.csv')
        st.dataframe(df_preprocessed.head(10))
