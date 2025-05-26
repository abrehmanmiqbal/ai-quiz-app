# model_training.py
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Sample difficulty-labeled data (replace with real training set for production)
data = {
    "question": [
        "What is 2+2?", 
        "Explain recursion in programming.",
        "Describe the use of normalization in neural networks.",
        "What is the capital of France?",
        "How does a blockchain maintain integrity?"
    ],
    "difficulty": ["easy", "medium", "hard", "easy", "hard"]
}

df = pd.DataFrame(data)

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
y = df["difficulty"]

# Train ANN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
clf.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(clf, "ann_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
