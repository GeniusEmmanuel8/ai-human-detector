import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── NLTK SETUP ────────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = [t for t in text.split() if len(t)>1 and t not in stop_words]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)

@st.cache_resource
def load_models():
    vect = load("models/tfidf_vectorizer.pkl")
    svm  = load("models/svm_model.pkl")
    dt   = load("models/decision_tree_model.pkl")
    return vect, {"SVM": svm, "Decision Tree": dt}

vectorizer, models = load_models()

st.title("🤖 AI vs Human Text Detector")

txt = st.text_area("Paste your text here:")
f    = st.file_uploader("…or upload a PDF/DOCX/TXT", type=["pdf","docx","txt"])

def extract_text(file) -> str:
    if file.name.lower().endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if file.name.lower().endswith(".docx"):
        return docx2txt.process(file)
    if file.name.lower().endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

content = txt or (extract_text(f) if f else "")
if not content:
    st.info("Please paste text or upload a file above.")
    st.stop()

choice = st.selectbox("Choose a model", list(models.keys()))
if st.button("Predict"):
    cleaned = preprocess_text(content)
    if not cleaned:
        st.error("Nothing left after preprocessing.")
        st.stop()
    X_vec = vectorizer.transform([cleaned])
    clf   = models[choice]
    pred  = clf.predict(X_vec)[0]
    st.subheader("Prediction:")
    st.write("🧠 AI-Written" if pred==1 else "👤 Human-Written")
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X_vec)[0]
        st.write(f"Confidence: {max(p):.2%}")
        st.write(f"Human: {p[0]:.2%}   AI: {p[1]:.2%}")
    with st.expander("Show cleaned text"):
        st.write(cleaned)
