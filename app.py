import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fungsi membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk memuat dan melatih model
@st.cache
def load_and_train_model():
    data = pd.read_excel('dataset.xlsx')
    data['clean_text'] = data['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(data['clean_text']).toarray()
    y = data['label']
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model

# Muat model
vectorizer, model = load_and_train_model()

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

