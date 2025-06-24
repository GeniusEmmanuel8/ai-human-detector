import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── NLTK SETUP ────────────────────────────────────────────────────────────────
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_and_filter(text: str) -> list[str]:
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in stop_words]

def lemmatize_text(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(tok) for tok in tokens]

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = tokenize_and_filter(text)
    lemmas = lemmatize_text(tokens)
    return " ".join(lemmas)

# ─── LOAD YOUR SAVED VECTORIZER & MODELS ───────────────────────────────────────
vectorizer = load("models/tfidf_vectorizer.pkl")

models = {
    "SVM": load("models/svm_model.pkl"),
    "Decision Tree": load("models/decision_tree_model.pkl"),
    # AdaBoost removed per homework instructions
}

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.title("🤖 AI vs Human Text Detector")

text_input = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader(
    "Or upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
)

def extract_text(file) -> str:
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            pages = [p.extract_text() for p in pdf.pages]
            return "\n".join([t for t in pages if t])
    if file.name.endswith(".docx"):
        return docx2txt.process(file)
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

final_text = text_input or (extract_text(uploaded_file) if uploaded_file else "")

if final_text:
    model_name = st.selectbox("Choose a model", list(models.keys()))
    if st.button("Predict"):
        # 1) Preprocess
        cleaned = preprocess_text(final_text)
        # 2) Vectorize
        X = vectorizer.transform([cleaned])
        # 3) Predict
        clf = models[model_name]
        pred = clf.predict(X)[0]
        st.subheader("Prediction:")
        st.write("🧠 AI-Written" if pred == 1 else "👤 Human-Written")
        # 4) Confidence (if available)
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X)[0][1]
            st.write(f"Confidence: {prob:.2f}")
