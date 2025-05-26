# train_ann.py
import joblib
import random
from quiz_generator import generate_mcqs
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Generate data
topic = "Machine Learning"
mcqs = generate_mcqs(topic)
questions = [q for q in mcqs]
labels = [random.choice(["easy", "medium", "hard"]) for _ in mcqs]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train ANN
clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000)
clf.fit(X, y)

# Save all components
joblib.dump(clf, "ann_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ ANN model, vectorizer, and label encoder saved.")
