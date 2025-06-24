import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── NLTK SETUP ────────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

# ─── LOAD VECTOR & MODELS ─────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    vect = load("models/tfidf_vectorizer.pkl")
    svm  = load("models/svm_model.pkl")
    dt   = load("models/decision_tree_model.pkl")
    return vect, {"SVM": svm, "Decision Tree": dt}

vectorizer, models = load_assets()

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.title("🤖 AI vs Human Text Detector")

text_input    = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader("Or upload a PDF, DOCX, or TXT", type=["pdf","docx","txt"])

def extract_text(file) -> str:
    try:
        if file.name.lower().endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        if file.name.lower().endswith(".docx"):
            # docx2txt.process expects a file path or file-like object
            # Streamlit's uploader provides a file-like object, but sometimes needs to be reset
            file.seek(0)
            return docx2txt.process(file)
        if file.name.lower().endswith(".txt"):
            file.seek(0)
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""
    return ""

final_text = text_input or (extract_text(uploaded_file) if uploaded_file else "")

# Ensure final_text is always a string
if not isinstance(final_text, str):
    final_text = str(final_text)

if not final_text.strip():
    st.info("Enter text or upload a file above to begin.")
    st.stop()

model_name = st.selectbox("Choose a model", list(models.keys()))

if st.button("Predict"):
    # 1) Preprocess exactly as during training
    cleaned = preprocess_text(final_text)
    if not isinstance(cleaned, str):
        st.error("Preprocessing did not return a string.")
        st.stop()
    if not cleaned.strip():
        st.error("Nothing left after preprocessing—try different text.")
        st.stop()

    # 2) Vectorize and predict
    try:
        X_vec = vectorizer.transform([cleaned])
    except Exception as e:
        st.error(f"Vectorization error: {e}")
        st.stop()
    clf   = models[model_name]

    pred  = clf.predict(X_vec)[0]
    st.subheader("Prediction:")
    st.write("🧠 **AI-Written**"   if pred == 1 else
             "👤 **Human-Written**")

    # 3) Show confidence if available
    if hasattr(clf, "predict_proba"):
        proba      = clf.predict_proba(X_vec)[0]
        confidence = max(proba)
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write(f"👤 Human: {proba[0]:.2%}")
        st.write(f"🤖 AI:    {proba[1]:.2%}")

    # 4) (optional) show the cleaned text
    with st.expander("Show preprocessed text"):
        st.write(cleaned)
