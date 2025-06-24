import streamlit as st
import pdfplumber
import docx2txt
from joblib import load
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── NLTK SETUP ────────────────────────────────────────────────────────────────
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Error setting up NLTK: {e}")
    st.stop()

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_and_filter(text: str) -> list[str]:
    """Tokenize and filter out stopwords and short tokens"""
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in stop_words]

def lemmatize_text(tokens: list[str]) -> list[str]:
    """Lemmatize tokens"""
    return [lemmatizer.lemmatize(tok) for tok in tokens]

def preprocess_text(text: str) -> str:
    """Complete text preprocessing pipeline"""
    if not isinstance(text, str):
        text = str(text)
    
    text = clean_text(text)
    tokens = tokenize_and_filter(text)
    lemmas = lemmatize_text(tokens)
    return " ".join(lemmas)

# ─── LOAD YOUR SAVED VECTORIZER & MODELS ───────────────────────────────────────
@st.cache_resource
def load_models():
    """Load vectorizer and models with error handling"""
    try:
        vectorizer = load("models/tfidf_vectorizer.pkl")
        models = {
            "SVM": load("models/svm_model.pkl"),
            "Decision Tree": load("models/decision_tree_model.pkl"),
        }
        return vectorizer, models
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure the models/ directory contains the required .pkl files")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

vectorizer, models = load_models()

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.title("🤖 AI vs Human Text Detector")

text_input = st.text_area("Paste your text here:")
uploaded_file = st.file_uploader(
    "Or upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
)

def extract_text(file) -> str:
    """Extract text from uploaded file"""
    try:
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                return "\n".join(pages)
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return ""
    return ""

# Get final text
final_text = ""
if text_input:
    final_text = text_input
elif uploaded_file:
    final_text = extract_text(uploaded_file)

if final_text:
    model_name = st.selectbox("Choose a model", list(models.keys()))
    
    if st.button("Predict"):
        try:
            # Debug: Show original text length
            st.write(f"Original text length: {len(final_text)} characters")
            
            # 1) Preprocess
            cleaned = preprocess_text(final_text)
            
            # Debug: Show cleaned text length
            st.write(f"Cleaned text length: {len(cleaned)} characters")
            
            # Ensure we have some text after preprocessing
            if not cleaned or len(cleaned.strip()) == 0:
                st.error("No text remaining after preprocessing. Please try with different text.")
                st.stop()
            
            # 2) Vectorize - Make sure we pass a list of strings
            X = vectorizer.transform([cleaned])  # Note: cleaned should be a string
            
            # Debug: Show vectorization result
            st.write(f"Vectorization successful. Shape: {X.shape}")
            
            # 3) Predict
            clf = models[model_name]
            pred = clf.predict(X)[0]
            
            # 4) Display results
            st.subheader("Prediction:")
            if pred == 1:
                st.write("🧠 **AI-Written**")
            else:
                st.write("👤 **Human-Written**")
            
            # 5) Confidence (if available)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)[0]
                confidence = max(proba)
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Show probability for each class
                st.write(f"Human probability: {proba[0]:.2%}")
                st.write(f"AI probability: {proba[1]:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.error("Please check your input text and try again.")
            
            # Debug information
            st.write("Debug info:")
            st.write(f"Text type: {type(final_text)}")
            st.write(f"Cleaned text type: {type(cleaned) if 'cleaned' in locals() else 'Not created'}")
else:
    st.info("Please enter some text or upload a file to analyze.")
