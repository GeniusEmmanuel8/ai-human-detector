import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# ——— Download NLTK data (once) ———
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ——— Preprocessing setup ———
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)              # remove numbers
    text = re.sub(r'\w*\d\w*', '', text)          # remove any mixed alphanumeric
    text = re.sub(r'[^\w\s]', '', text)           # remove punctuation
    return text

def tokenize_and_filter(text: str) -> list[str]:
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in stop_words]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_text(text: str) -> str:
    txt = clean_text(text)
    toks = tokenize_and_filter(txt)
    lemmas = lemmatize_tokens(toks)
    return ' '.join(lemmas)

# ——— Load vectorizer + models ———
vectorizer = load("models/tfidf_vectorizer.pkl")
models = {
    "SVM":            load("models/svm_model.pkl"),
    "Decision Tree": load("models/decision_tree_model.pkl"),
    "AdaBoost":      load("models/adaboost_model.pkl"),
}

# ——— Streamlit UI ———
st.title("🤖 AI vs Human Text Detector")

text_input    = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader("Or upload a PDF, DOCX, or TXT", type=["pdf","docx","txt"])

def extract_text(file) -> str:
    if file.name.lower().endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if file.name.lower().endswith(".docx"):
        return docx2txt.process(file)
    if file.name.lower().endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

# Determine source text
final_text = text_input or (extract_text(uploaded_file) if uploaded_file else "")

if final_text:
    model_choice = st.selectbox("Choose a model", list(models.keys()))
    if st.button("Predict"):
        # 1) preprocess → 2) vectorize → 3) predict
        cleaned = preprocess_text(final_text)
        X_vec   = vectorizer.transform([cleaned])
        m       = models[model_choice]

        pred = m.predict(X_vec)[0]
        prob = m.predict_proba(X_vec)[0][1] if hasattr(m, "predict_proba") else None

        st.subheader("Prediction:")
        st.write("🧠 **AI-Written**" if pred == 1 else "👤 **Human-Written**")
        if prob is not None:
            st.write(f"Confidence: **{prob:.2f}**")
