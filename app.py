import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Fungsi membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk memuat dan melatih model
@st.cache_data
def load_and_train_model():
    try:
        # Memuat dataset
        file_path = 'static/cleaned_dataset-_1_.xlsx'
        if not os.path.exists(file_path):
            st.error(f"File tidak ditemukan: {file_path}")
            return None, None

        data = pd.read_excel(file_path)
        data['clean_text'] = data['text'].apply(clean_text)

        # Membuat fitur TF-IDF dan melatih model
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(data['clean_text']).toarray()
        y = data['label']
        model = LogisticRegression()
        model.fit(X, y)

        return vectorizer, model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau melatih model: {e}")
        return None, None

# Muat model
vectorizer, model = load_and_train_model()

if vectorizer and model:
    # Streamlit UI
    st.title("Sentiment Analysis App")
    input_text = st.text_input("Masukkan teks untuk dianalisis:")
    
    if st.button("Analisis"):
        if input_text:
            clean_input = clean_text(input_text)
            input_vectorized = vectorizer.transform([clean_input]).toarray()
            prediction = model.predict(input_vectorized)[0]
            st.write(f"Prediksi Sentimen: {prediction}")
        else:
            st.write("Harap masukkan teks!")
else:
    st.write("Model gagal dimuat. Cek file atau perbaiki masalah lainnya.")
