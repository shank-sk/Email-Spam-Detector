import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import joblib
from pathlib import Path

# Download required NLTK data (no-op if already present)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Helper: safe model loading
MODEL_PATH = Path("spam_classifier.pkl")
VECT_PATH = Path("tfidf_vectorizer.pkl")

clf = None
vectorizer = None
try:
    if MODEL_PATH.exists() and VECT_PATH.exists():
        clf = joblib.load(str(MODEL_PATH))
        vectorizer = joblib.load(str(VECT_PATH))
    else:
        st.warning("Model or vectorizer not found. Please ensure 'spam_classifier.pkl' and 'tfidf_vectorizer.pkl' are present in the app directory.")
except Exception as e:
    st.error(f"Error loading model files: {e}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Basic preprocessing: tokenize, lowercase, remove stopwords/punctuation, lemmatize."""
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    lemmatized = []
    for t in tokens:
        try:
            lm = lemmatizer.lemmatize(t)
        except LookupError:
            # WordNet data not available, skip lemmatization
            lm = t
        except Exception:
            lm = t
        lemmatized.append(lm)
    return " ".join(lemmatized)


# ---- Streamlit UI ----
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

st.title("📧 Spam Email Detection")
st.markdown("Detect whether an email is Spam or Ham (not spam). Enter text, upload a file, or try one of the example messages.")

# Sidebar for examples and options
with st.sidebar:
    st.header("Try examples")
    example = st.selectbox("Choose an example", ["-- pick an example --",
                                                    "Win a free iPhone now! Click here",
                                                    "Meeting agenda for tomorrow",
                                                    "Your account has been suspended, verify now",
                                                    "Monthly newsletter from our team"]) 
    st.markdown("---")
    st.header("Input options")
    uploaded_file = st.file_uploader("Upload a .txt email file", type=["txt"])
    uploaded_text = ""
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            # try utf-8 then fall back to latin-1
            try:
                uploaded_text = raw.decode('utf-8')
            except Exception:
                uploaded_text = raw.decode('latin-1')
            st.info(f"Loaded file: {uploaded_file.name} ({len(raw)} bytes)")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
    st.markdown("---")
    st.caption("Model files should be in the app folder: spam_classifier.pkl and tfidf_vectorizer.pkl")


col1, col2 = st.columns([3, 2])

with col1:
    with st.form(key="email_form"):
        # Determine initial value: uploaded file takes precedence, then selected example
        initial_value = uploaded_text if uploaded_text else (example if example and example != "-- pick an example --" else "")
        email_text = st.text_area("Email text", height=250, value=initial_value)
        submitted = st.form_submit_button("Predict")
        clear = st.form_submit_button("Clear")

with col2:
    st.markdown("### Result")
    result_placeholder = st.empty()
    st.markdown("### Details")
    details = st.empty()

if 'email_text' not in locals():
    email_text = ""

if clear:
    # simple clear behaviour
    email_text = ""
    result_placeholder.info("Input cleared. Enter text or select an example.")

if submitted:
    if not email_text:
        result_placeholder.warning("Please enter some email text or choose/upload an example.")
    elif clf is None or vectorizer is None:
        result_placeholder.error("Model not loaded. Prediction not available.")
    else:
        processed_email = preprocess(email_text)
        try:
            email_vec = vectorizer.transform([processed_email])
            pred = clf.predict(email_vec)[0]
            probs = None
            # Try to get probability if available
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(email_vec)[0]

            label = "Spam" if int(pred) == 1 else "Ham"
            # colored result
            if label == "Spam":
                result_placeholder.markdown(f"<div style='padding:12px;border-radius:6px;background:#ffdddd;color:#900;font-weight:700'>Prediction: {label}</div>", unsafe_allow_html=True)
            else:
                result_placeholder.markdown(f"<div style='padding:12px;border-radius:6px;background:#ddffea;color:#065;font-weight:700'>Prediction: {label}</div>", unsafe_allow_html=True)

            # show probability if available
            if probs is not None:
                spam_prob = probs[1] if len(probs) > 1 else probs[0]
                details.metric("Spam probability", f"{spam_prob*100:.2f}%")
            else:
                details.info("Model does not expose probability scores.")

            st.markdown("---")
            st.subheader("Preprocessed text")
            st.code(processed_email)
        except Exception as e:
            result_placeholder.error(f"Prediction failed: {e}")

