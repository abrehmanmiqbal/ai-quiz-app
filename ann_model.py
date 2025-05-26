# ann_model.py
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ann_model.py
import joblib

def load_ann_model():
    model = joblib.load("ann_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder
