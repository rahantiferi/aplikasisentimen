from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json

# Inisialisasi Flask
app = Flask(__name__)

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text


# Muat dan latih model
def load_and_train_model():
    try:
        data = pd.read_excel('uploads/dataset.xlsx')  # Ganti dengan nama dataset Anda
        data = data.dropna(subset=['text', 'label'])  # Pastikan kolom text dan label ada

        # Membersihkan teks
        data['clean_text'] = data['text'].apply(clean_text)

        # Ekstraksi fitur dengan TF-IDF
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(data['clean_text']).toarray()
        y = data['label']

        # Latih model
        model = LogisticRegression()
        model.fit(X, y)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading or training model: {e}")
        return None, None

# Muat model dan vectorizer
vectorizer, model = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        clean_input = clean_text(input_text)
        input_vectorized = vectorizer.transform([clean_input]).toarray()

        # Periksa apakah model telah dimuat
        if model is None or vectorizer is None:
            return "Model tidak terlatih atau error memuat model", 500
        
        prediction = model.predict(input_vectorized)[0]

        # Pastikan nilai yang dikirim ke template adalah valid
        sentiment = prediction if prediction is not None else 'Tidak Diketahui'
        report = {'sentiment': sentiment}  # Contoh report dalam format dictionary

        # Mengonversi report ke JSON agar dapat diterima template
        try:
            report_json = json.dumps(report)
        except TypeError:
            report_json = '{}'

        return render_template('result.html', text=input_text, sentiment=prediction, report=report_json)

if __name__ == '__main__':
    app.run(debug=True)
