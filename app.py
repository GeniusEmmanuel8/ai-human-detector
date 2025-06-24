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
    """Complete text preprocessing pipeline - same as training"""
    if not isinstance(text, str):
        text = str(text)
    
    text = clean_text(text)
    tokens = tokenize_and_filter(text)
    lemmas = lemmatize_text(tokens)
    return " ".join(lemmas)

# ─── LOAD YOUR SAVED PIPELINE MODELS ───────────────────────────────────────────
@st.cache_resource
def load_pipeline_models():
    """Load complete pipeline models"""
    try:
        # Load the complete pipelines (not individual components)
        models = {
            "SVM": load("models/svm_pipeline.pkl"),  # or whatever you named it
            "Decision Tree": load("models/dt_pipeline.pkl"),  # or whatever you named it
        }
        
        st.success("✅ Pipeline models loaded successfully!")
        return models
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Expected files: svm_pipeline.pkl, dt_pipeline.pkl")
        st.error("Make sure to save your models like this:")
        st.code("""
# In your training script:
from joblib import dump
dump(svm_grid.best_estimator_, 'models/svm_pipeline.pkl')
dump(dt_grid.best_estimator_, 'models/dt_pipeline.pkl')
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

models = load_pipeline_models()

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
            # Debug info
            st.write(f"Original text length: {len(final_text)} characters")
            
            # Preprocess the text exactly as done during training
            preprocessed_text = preprocess_text(final_text)
            
            st.write(f"Preprocessed text length: {len(preprocessed_text)} characters")
            
            # Ensure we have some text after preprocessing
            if not preprocessed_text or len(preprocessed_text.strip()) == 0:
                st.error("No text remaining after preprocessing. Please try with different text.")
                st.stop()
            
            # Use the complete pipeline - it handles vectorization AND prediction
            pipeline = models[model_name]
            
            # The pipeline expects the same format as training (preprocessed text)
            prediction = pipeline.predict([preprocessed_text])[0]
            
            # Display results
            st.subheader("Prediction:")
            if prediction == 1:
                st.write("🧠 **AI-Written**")
            else:
                st.write("👤 **Human-Written**")
            
            # Get confidence scores
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba([preprocessed_text])[0]
                confidence = max(proba)
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Show probability for each class
                st.write(f"Human probability: {proba[0]:.2%}")
                st.write(f"AI probability: {proba[1]:.2%}")
            
            # Debug: Show what was actually processed
            with st.expander("Debug Info"):
                st.write("**Preprocessed text (first 200 chars):**")
                st.write(preprocessed_text[:200])
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
            # Enhanced debug information
            st.write("=== DEBUG INFORMATION ===")
            st.write(f"Final text type: {type(final_text)}")
            if 'preprocessed_text' in locals():
                st.write(f"Preprocessed text type: {type(preprocessed_text)}")
                st.write(f"Preprocessed sample: {repr(preprocessed_text[:100])}")
            
            import traceback
            st.code(traceback.format_exc())
            
else:
    st.info("Please enter some text or upload a file to analyze.")
    
    # Show instructions
    st.markdown("""
    ### Instructions:
    1. Either paste text directly into the text area above
    2. Or upload a PDF, DOCX, or TXT file
    3. Choose your preferred model (SVM or Decision Tree)
    4. Click "Predict" to see if the text was likely written by AI or human
    
    ### Expected Model Files:
    Make sure your `models/` directory contains:
    - `svm_pipeline.pkl` - Complete SVM pipeline
    - `dt_pipeline.pkl` - Complete Decision Tree pipeline
    
    ### How to save your models correctly:
    ```python
    from joblib import dump
    
    # After training your GridSearchCV
    dump(svm_grid.best_estimator_, 'models/svm_pipeline.pkl')
    dump(dt_grid.best_estimator_, 'models/dt_pipeline.pkl')
    ```
    """)
