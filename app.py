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
    """Lowercase, strip numbers/punctuation."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords, lemmatize."""
    text = clean_text(text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in stop_words]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)

@st.cache_resource
def load_models():
    """Load vectorizer + all three classifiers."""
    vect = load("models/tfidf_vectorizer.pkl")
    svm  = load("models/svm_model.pkl")
    dt   = load("models/decision_tree_model.pkl")
    ada  = load("models/adaboost_model.pkl")
    return vect, {
        "SVM": svm,
        "Decision Tree": dt,
        "Adaboost": ada
    }

vectorizer, models = load_models()

st.title("🤖 AI vs Human Text Detector")

# --- user input ---
txt = st.text_area("Paste your text here:")
f   = st.file_uploader("…or upload a PDF, DOCX, or TXT", type=["pdf","docx","txt"])

def extract_text(file) -> str:
    """Turn uploaded file into one big string."""
    try:
        name = file.name.lower()
        if name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                return str(text) if text else ""
        if name.endswith(".docx"):
            file.seek(0)  # Reset file pointer
            text = docx2txt.process(file)
            return str(text) if text else ""
        if name.endswith(".txt"):
            file.seek(0)  # Reset file pointer
            text = file.read().decode("utf-8")
            return str(text) if text else ""
        return ""
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return ""

# decide which to use
content = txt or (extract_text(f) if f else "")
if not isinstance(content, str):
    content = str(content) if content else ""

if not content or not content.strip():
    st.info("Please paste some text or upload a file above.")
    st.stop()

# model selector + predict
choice = st.selectbox("Choose a model", list(models.keys()))
if st.button("Predict"):
    # Ensure content is a string
    if not isinstance(content, str):
        st.error(f"Content is not a string. Type: {type(content)}")
        st.stop()
    
    cleaned = preprocess_text(content)
    
    # Check if cleaned is a string and not empty
    if not isinstance(cleaned, str):
        st.error(f"Preprocessing did not return a string. Type: {type(cleaned)}")
        st.stop()
    
    if not cleaned or not cleaned.strip():
        st.error("Nothing left after preprocessing → try different text.")
        st.stop()

    # Debug: show the cleaned text length and first few characters
    st.write(f"Debug: Cleaned text length: {len(cleaned)}")
    st.write(f"Debug: First 50 chars: {cleaned[:50]}")

    clf   = models[choice]
    try:
        pred  = clf.predict([cleaned])[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error(f"Cleaned text type: {type(cleaned)}")
        st.error(f"Cleaned text value: {repr(cleaned)}")
        st.stop()

    st.subheader("Prediction:")
    st.write("🧠 **AI-Written**" if pred == 1 else "👤 **Human-Written**")

    # show confidence if available
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba([cleaned])[0]
            st.write(f"**Confidence:** {max(p):.2%}")
            st.write(f"Human: {p[0]:.2%}   AI: {p[1]:.2%}")
        except Exception as e:
            st.write(f"Probability error: {e}")

    with st.expander("Show preprocessed text"):
        st.write(cleaned)
