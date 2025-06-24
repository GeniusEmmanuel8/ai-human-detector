import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline

# ─── NLTK SETUP ────────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
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
    if not isinstance(text, str):
        text = str(text)
    text = clean_text(text)
    tokens = tokenize_and_filter(text)
    lemmas = lemmatize_text(tokens)
    return " ".join(lemmas)

# ─── LOAD YOUR VECTORIZER + MODELS, BUILD PIPELINES ────────────────────────────
@st.cache_resource
def load_pipelines():
    try:
        vect = load("models/tfidf_vectorizer.pkl")
        svm = load("models/svm_model.pkl")
        dt  = load("models/decision_tree_model.pkl")
    except FileNotFoundError as e:
        st.error(f"Missing model file: {e}")
        st.stop()

    pipelines = {
        "SVM": Pipeline([("tfidf", vect), ("clf", svm)]),
        "Decision Tree": Pipeline([("tfidf", vect), ("clf", dt)])
    }
    return pipelines

models = load_pipelines()

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.title("🤖 AI vs Human Text Detector")

text_input = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader(
    "Or upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
)

def extract_text(file) -> str:
    try:
        if file.name.lower().endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        if file.name.lower().endswith(".docx"):
            return docx2txt.process(file)
        if file.name.lower().endswith(".txt"):
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return ""

final_text = text_input or (extract_text(uploaded_file) if uploaded_file else "")

if not final_text:
    st.info("Please enter text or upload a file above to analyze.")
    st.stop()

model_name = st.selectbox("Choose a model", list(models.keys()))

if st.button("Predict"):
    # 1) Preprocess
    pre = preprocess_text(final_text)
    if not pre.strip():
        st.error("Nothing left after preprocessing—try different text.")
        st.stop()

    # 2) Run through the pipeline
    pipe = models[model_name]
    pred = pipe.predict([pre])[0]

    st.subheader("Prediction:")
    st.write("🧠 **AI-Written**" if pred == 1 else "👤 **Human-Written**")

    # 3) Confidence
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba([pre])[0]
        st.write(f"**Confidence:** {max(probs):.2%}")
        st.write(f"Human: {probs[0]:.2%}, AI: {probs[1]:.2%}")

    # 4) Debug expander
    with st.expander("Show me the preprocessed text"):
        st.write(pre)
