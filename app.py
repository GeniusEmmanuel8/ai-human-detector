import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------- TEXT PREPROCESSING ---------------------
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_and_filter(text):
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in stop_words]

def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_and_filter(text)
    lemmas = lemmatize_text(tokens)
    return ' '.join(lemmas)

# --------------------- LOAD MODELS ---------------------
vectorizer = load("models/tfidf_vectorizer.pkl")
models = {
    "SVM": load("models/svm_model.pkl"),
    "Decision Tree": load("models/decision_tree_model.pkl"),
}

# --------------------- STREAMLIT UI ---------------------
st.title("🤖 AI vs Human Text Detector")

text_input = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader("Or upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

final_text = text_input or (extract_text(uploaded_file) if uploaded_file else "")

if final_text:
    model_choice = st.selectbox("Choose a model", list(models.keys()))
    if st.button("Predict"):
        cleaned = preprocess_text(final_text)
        X = vectorizer.transform([cleaned])
        model = models[model_choice]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
        st.subheader("Prediction:")
        st.write("🧠 AI-Written" if pred == 1 else "👤 Human-Written")
        if prob is not None:
            st.write(f"Confidence: {prob:.2f}")
