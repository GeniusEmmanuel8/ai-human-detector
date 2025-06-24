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
        
        # Debug: Print vectorizer info
        st.write("Vectorizer loaded successfully")
        st.write(f"Vectorizer type: {type(vectorizer)}")
        if hasattr(vectorizer, 'get_params'):
            params = vectorizer.get_params()
            st.write(f"Vectorizer preprocessor: {params.get('preprocessor', 'None')}")
            st.write(f"Vectorizer lowercase: {params.get('lowercase', 'None')}")
        
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
            # Debug: Show original text info
            st.write(f"Original text length: {len(final_text)} characters")
            st.write(f"Original text type: {type(final_text)}")
            st.write(f"First 100 chars: {repr(final_text[:100])}")
            
            # 1) Preprocess - Try different approaches
            
            # Option 1: Use your preprocessing
            cleaned = preprocess_text(final_text)
            st.write(f"After preprocessing - Length: {len(cleaned)}, Type: {type(cleaned)}")
            st.write(f"Cleaned sample: {repr(cleaned[:100])}")
            
            # Ensure we have some text after preprocessing
            if not cleaned or len(cleaned.strip()) == 0:
                st.error("No text remaining after preprocessing. Please try with different text.")
                st.stop()
            
            # Option 2: Try with minimal preprocessing for the vectorizer
            # Some vectorizers expect raw text and do their own preprocessing
            
            # Try both approaches
            st.write("Trying with preprocessed text...")
            try:
                X1 = vectorizer.transform([cleaned])
                st.success("✅ Preprocessed text worked!")
                X = X1
            except Exception as e1:
                st.write(f"❌ Preprocessed text failed: {e1}")
                st.write("Trying with raw text...")
                try:
                    # Try with just the original text
                    X2 = vectorizer.transform([final_text])
                    st.success("✅ Raw text worked!")
                    X = X2
                except Exception as e2:
                    st.write(f"❌ Raw text also failed: {e2}")
                    # Try with minimal cleaning
                    st.write("Trying with minimal cleaning...")
                    minimal_clean = re.sub(r'[^\w\s]', ' ', final_text.lower()).strip()
                    X3 = vectorizer.transform([minimal_clean])
                    st.success("✅ Minimal cleaning worked!")
                    X = X3
            
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
            
            # Enhanced debug information
            st.write("=== DEBUG INFORMATION ===")
            st.write(f"Final text type: {type(final_text)}")
            st.write(f"Final text length: {len(final_text) if hasattr(final_text, '__len__') else 'No length'}")
            if 'cleaned' in locals():
                st.write(f"Cleaned text type: {type(cleaned)}")
                st.write(f"Cleaned text length: {len(cleaned) if hasattr(cleaned, '__len__') else 'No length'}")
            
            # Try to see what the vectorizer expects
            if hasattr(vectorizer, 'get_params'):
                st.write(f"Vectorizer params: {vectorizer.get_params()}")
            
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Please enter some text or upload a file to analyze.")
