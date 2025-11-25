"""
MuToXGuard - Streamlit Web UI (Simple Version)
Multilingual Toxicity Detection Interface
"""
import streamlit as st
import joblib
import pandas as pd
from langdetect import detect_langs
import re
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

# Page config
st.set_page_config(
    page_title="MuToXGuard - Toxicity Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .toxic-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #ffc107;
    }
    .toxic {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
    }
    .stat-box {
        padding: 15px;
        border-radius: 8px;
        background: #f8f9fa;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = BASE_DIR / 'models' / 'classical' / 'logreg_toxic_only.joblib'
        vectorizer_path = BASE_DIR / 'models' / 'classical' / 'tfidf_vectorizer.joblib'
        
        # Debug: Show paths
        st.sidebar.write(f"BASE_DIR: {BASE_DIR}")
        st.sidebar.write(f"Model path exists: {model_path.exists()}")
        st.sidebar.write(f"Looking for model at: {model_path}")
        
        model = joblib.load(str(model_path))
        vectorizer = joblib.load(str(vectorizer_path))
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error(f"Current directory: {Path.cwd()}")
        st.error(f"Script directory: {BASE_DIR}")
        st.error(f"Files in BASE_DIR: {list(BASE_DIR.iterdir()) if BASE_DIR.exists() else 'DIR NOT FOUND'}")
        raise

def detect_language(text):
    """Detect language of text"""
    try:
        langs = detect_langs(text)
        if langs:
            return langs[0].lang.upper()
    except:
        pass
    return "UNKNOWN"

def analyze_text(text, model, vectorizer):
    """Analyze text for toxicity"""
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    
    return {
        'is_toxic': bool(pred),
        'confidence': float(prob),
        'severity': 'HIGH' if prob > 0.8 else 'MEDIUM' if prob > 0.5 else 'LOW'
    }

# Header
st.markdown('<h1 class="main-title">🛡️ MuToXGuard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multilingual Toxicity Detection System</p>', unsafe_allow_html=True)

# Load model
try:
    model, vectorizer = load_model()
    st.success("✓ Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **MuToXGuard** is an advanced toxicity detection system that supports:
    
    - 🌍 Multiple languages (EN, MS, mixed)
    - 🎯 Real-time analysis
    - 🔍 Confidence scoring
    - 📊 Detailed metrics
    """)
    
    st.header("📊 Model Info")
    st.write(f"""
    - **Algorithm**: Logistic Regression
    - **Features**: {len(vectorizer.vocabulary_):,}
    - **Accuracy**: 85%
    - **F1 Score**: 0.74
    """)
    
    st.header("🎨 Examples")
    if st.button("Try Example 1"):
        st.session_state.example = "You are awesome! Great work!"
    if st.button("Try Example 2"):
        st.session_state.example = "This is stupid and you're an idiot"
    if st.button("Try Example 3"):
        st.session_state.example = "Bodoh punya orang"

# Main input
st.header("📝 Enter Text to Analyze")

text_input = st.text_area(
    "Type or paste text here:",
    value=st.session_state.get('example', ''),
    height=150,
    placeholder="Enter any text in English or Malay..."
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.example = ''
    st.rerun()

# Analysis
if analyze_btn and text_input.strip():
    with st.spinner("Analyzing..."):
        # Detect language
        lang = detect_language(text_input)
        
        # Analyze
        result = analyze_text(text_input, model, vectorizer)
        
        # Display results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Main result box
        if result['is_toxic']:
            if result['severity'] == 'HIGH':
                st.markdown(f'<div class="toxic-box toxic">🔴 TOXIC CONTENT DETECTED - {result["severity"]} RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="toxic-box warning">⚠️ POTENTIALLY TOXIC - {result["severity"]} RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="toxic-box safe">✅ SAFE CONTENT</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Language", lang)
        
        with col2:
            confidence_pct = result['confidence'] * 100
            st.metric("Toxicity Score", f"{confidence_pct:.1f}%")
        
        with col3:
            st.metric("Severity Level", result['severity'])
        
        # Progress bar
        st.write("**Confidence Visualization:**")
        st.progress(result['confidence'])
        
        # Text stats
        st.markdown("---")
        st.subheader("📈 Text Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        words = text_input.split()
        chars = len(text_input)
        
        with col1:
            st.markdown(f'<div class="stat-box">Words: <b>{len(words)}</b></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box">Characters: <b>{chars}</b></div>', unsafe_allow_html=True)
        with col3:
            has_caps = any(c.isupper() for c in text_input)
            st.markdown(f'<div class="stat-box">Caps: <b>{"Yes" if has_caps else "No"}</b></div>', unsafe_allow_html=True)
        with col4:
            has_special = bool(re.search(r'[!?@#$%^&*]', text_input))
            st.markdown(f'<div class="stat-box">Special Chars: <b>{"Yes" if has_special else "No"}</b></div>', unsafe_allow_html=True)
        
        # Detailed explanation
        with st.expander("ℹ️ Understanding the Results"):
            st.write("""
            **How to interpret:**
            
            - **Toxicity Score**: Probability (0-100%) that the text is toxic
            - **Severity Levels**:
              - LOW: < 50% - Likely safe
              - MEDIUM: 50-80% - Potentially concerning
              - HIGH: > 80% - Highly toxic
            
            **Note**: This is an automated system and may not be 100% accurate. 
            Use human judgment for critical decisions.
            """)

elif analyze_btn:
    st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>MuToXGuard v1.0 | Built with ❤️ using Streamlit</p>
    <p>Supporting multilingual toxicity detection for safer online communities</p>
</div>
""", unsafe_allow_html=True)
