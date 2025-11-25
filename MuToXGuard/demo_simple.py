"""
MuToXGuard Demo - Simple Inference
Quick demonstration of toxicity detection
"""
import joblib
import pandas as pd

# Load model
print("Loading model...")
model = joblib.load('models/classical/logreg_toxic_only.joblib')
vectorizer = joblib.load('models/classical/tfidf_vectorizer.joblib')

print("\n" + "="*80)
print("MuToXGuard - Multilingual Toxicity Detection Demo")
print("="*80)

# Test examples
examples = [
    "You are awesome! Great work!",
    "This is stupid and you're an idiot",
    "I love this product, highly recommend it",
    "Go kill yourself you worthless piece of trash",
    "Apa khabar? Saya suka makanan ini",  # Malay: How are you? I like this food
    "Bodoh punya orang",  # Malay: Stupid person
]

print("\nTesting examples:\n")
for i, text in enumerate(examples, 1):
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    
    # Display
    status = "🔴 TOXIC" if pred == 1 else "🟢 SAFE"
    print(f"{i}. {status} (confidence: {prob:.2%})")
    print(f"   Text: {text}")
    print()

print("="*80)
print("\nModel Info:")
print(f"- Classifier: Logistic Regression")
print(f"- Features: {len(vectorizer.vocabulary_):,} TF-IDF features")
print(f"- Validation Accuracy: 85%")
print(f"- Validation F1 (toxic): 0.74")
print("="*80)
