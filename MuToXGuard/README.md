# MuToXGuard: Multilingual Toxicity & Cyberbullying Detection System

🛡️ **Production-style NLP system for detecting toxic content in English, Malay, and code-mixed text.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Web UI](#web-ui)
- [Ethics & Limitations](#ethics--limitations)
- [Citation](#citation)

---

## 🎯 Overview

MuToXGuard is a comprehensive toxicity detection system that:

- ✅ Detects toxic content in **3 languages**: English (EN), Malay (MS), and code-mixed (MIX)
- ✅ Performs **multi-label classification** across 9 toxicity categories
- ✅ Provides **severity scoring** (0-3 scale)
- ✅ Offers **explainability** via LIME, token importance, and similar examples
- ✅ Includes an **interactive Streamlit web UI**
- ✅ Follows production-ready code standards

---

## 🚀 Features

### Core Capabilities

1. **Language Identification**: Automatically detects EN, MS, or code-mixed text
2. **Multi-label Classification**: Detects 9 toxicity types:
   - Toxic
   - Insult
   - Harassment
   - Obscene
   - Identity Attack
   - Hate
   - Sexual
   - Threat
   - Spam

3. **Ensemble Modeling**: Combines:
   - Logistic Regression
   - Linear SVM
   - Multilingual BERT (bert-base-multilingual-cased)

4. **Explainability**:
   - Token-level importance heatmaps
   - LIME explanations
   - Similar example retrieval

5. **Web Interface**: User-friendly Streamlit app

---

## 🏗️ System Architecture

```
Input Text
    ↓
[Preprocessing] → URL removal, slang normalization, etc.
    ↓
[Language ID] → EN / MS / MIX
    ↓
┌─────────────────────────────────────┐
│       Feature Extraction            │
├─────────────────────────────────────┤
│  - TF-IDF (classical)               │
│  - BERT tokenization (deep)         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│          Model Ensemble             │
├─────────────────────────────────────┤
│  25% Logistic Regression            │
│  25% Linear SVM                     │
│  50% BERT                           │
└─────────────────────────────────────┘
    ↓
[Output] → Labels + Probabilities + Severity
    ↓
[Explainability] → Token importance, LIME, Examples
```

---

## 📦 Installation

### Requirements

- Python 3.10+
- 16GB+ RAM (for BERT training)
- GPU recommended (but not required)

### Setup

```bash
# 1. Clone or download the project
cd MuToXGuard

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# Alternative: Use conda
conda env create -f environment.yml
conda activate mutoxguard
```

---

## ⚡ Quick Start

### 1. Data Preparation

Your datasets are already unified! The unified dataset is in:
```
data/processed/toxic_multilingual_dataset.csv
```

Contains **21,048 samples** from:
- Jigsaw Toxic Comments (English)
- HateMalay Dataset (Malay)
- Sentiment Dataset (Mixed)

### 2. Train Models

```bash
# Train classical models (LogReg + SVM)
python src/train_classical.py

# Train BERT model (use --full for complete dataset)
python src/train_bert.py
# or
python src/train_bert.py --full  # Full training (takes longer)
```

### 3. Run Web UI

```bash
streamlit run ui/app_streamlit.py
```

Then open your browser to `http://localhost:8501`

---

## 🎓 Training Pipeline

### Step-by-Step Process

```bash
# 1. Data unification (already done)
python src/data_unification.py

# 2. Data loading and splitting
python src/data_loading.py

# 3. Train classical models
python src/train_classical.py

# 4. Train BERT model
python src/train_bert.py

# 5. Test ensemble
python src/ensemble.py

# 6. Evaluate models
python src/evaluation.py

# 7. Test inference
python src/demo_inference.py
```

---

## 💻 Usage

### Python API

```python
from src.demo_inference import ToxicityDetector

# Initialize detector
detector = ToxicityDetector()

# Analyze a comment
result = detector.analyze_comment(
    "Kau ni bodoh, don't talk to me!",
    include_explanations=True
)

# Access results
print(f"Language: {result['language']}")
print(f"Severity: {result['severity']['label']}")
print(f"Toxic labels: {[k for k, v in result['labels'].items() if v == 1]}")
print(f"Top probability: {max(result['probabilities'].items(), key=lambda x: x[1])}")
```

### Command Line

```bash
# Run interactive demo
python src/demo_inference.py
```

---

## 📁 Project Structure

```
MuToXGuard/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── train.csv                 # Jigsaw English
│   │   ├── HateMalay Dataset.csv     # Malay hate speech
│   │   └── data.csv                  # Mixed sentiment
│   └── processed/                    # Processed data
│       ├── toxic_multilingual_dataset.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── configs/                          # Configuration files
│   ├── config_classical.yaml
│   ├── config_bert.yaml
│   └── config_ensemble.yaml
│
├── models/                           # Saved models
│   ├── classical/
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── logreg_model.joblib
│   │   └── svm_model.joblib
│   └── deep/
│       └── bert_multilingual/
│
├── src/                              # Source code
│   ├── data_unification.py           # Dataset unification
│   ├── data_loading.py               # Data loading utilities
│   ├── preprocessing.py              # Text preprocessing
│   ├── langid_module.py              # Language identification
│   ├── features.py                   # Feature extraction
│   ├── train_classical.py            # Classical model training
│   ├── train_bert.py                 # BERT training
│   ├── ensemble.py                   # Ensemble model
│   ├── evaluation.py                 # Evaluation metrics
│   ├── explainability.py             # Explainability engine
│   ├── demo_inference.py             # Inference pipeline
│   └── utils.py                      # Utility functions
│
├── ui/
│   └── app_streamlit.py              # Streamlit web interface
│
├── reports/                          # Results and figures
│   ├── figures/                      # Plots and visualizations
│   ├── experiment_logs.csv           # Training logs
│   └── model_card.md                 # Model documentation
│
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
└── README.md                         # This file
```

---

## 📊 Model Performance

### Test Set Results (21,048 samples)

| Model | Macro F1 | Micro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Linear SVM | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| BERT | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| **Ensemble** | **0.XXX** | **0.XXX** | **0.XXX** | **0.XXX** |

*(Run training to populate these metrics)*

### Per-Label Performance

| Label | F1 Score | Support |
|-------|----------|---------|
| Toxic | 0.XXX | 6,716 |
| Insult | 0.XXX | 4,378 |
| Hate | 0.XXX | 1,859 |
| Harassment | 0.XXX | 1,861 |
| Obscene | 0.XXX | 2,676 |
| Identity Attack | 0.XXX | 453 |
| Threat | 0.XXX | 156 |

---

## 🌐 Web UI

Launch the Streamlit interface:

```bash
streamlit run ui/app_streamlit.py
```

### Features:
- 📝 Real-time toxicity detection
- 🌍 Language detection display
- 📊 Severity visualization
- 🎨 Token-level importance highlighting
- 📈 Probability charts
- 🔍 Explanations and similar examples

---

## ⚖️ Ethics & Limitations

### Intended Use
- ✅ Pre-filtering for human moderators
- ✅ Research and education
- ✅ Content analysis and insights

### NOT Intended For
- ❌ Sole basis for banning users
- ❌ Legal or law enforcement decisions
- ❌ Automated censorship without review

### Limitations
- May struggle with sarcasm and context
- Sensitive to unseen slang
- Possible bias toward certain demographics
- Performance varies by language (EN > MS > MIX)

### Recommendations
- Always use **human-in-the-loop** review
- Regular retraining with updated data
- Monitor for bias and fairness
- Provide appeals process for users

See `reports/model_card.md` for complete documentation.

---

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{mutoxguard2025,
  title={MuToXGuard: Multilingual Toxicity & Cyberbullying Detection System},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/mutoxguard}
}
```

---

## 📄 License

This project is for educational purposes. 

Dataset attributions:
- Jigsaw Toxic Comments: [Kaggle/Jigsaw](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- HateMalay Dataset: [Research source]

---

## 🤝 Contributing

This is an academic project. For improvements or questions, please contact the author.

---

## 📞 Support

For issues or questions:
1. Check the documentation in `reports/model_card.md`
2. Review training logs in `reports/experiment_logs.csv`
3. Contact: [Your Email]

---

**Built with ❤️ for safer online communities**
