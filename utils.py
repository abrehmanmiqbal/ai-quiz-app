# utils.py
import joblib

clf = joblib.load("ann_model.pkl")            # ANN classifier
vectorizer = joblib.load("vectorizer.pkl")    # Vectorizer used for questions

def classify_question_difficulty(question):
    X = vectorizer.transform([question])
    prediction = clf.predict(X)
    return prediction[0]
