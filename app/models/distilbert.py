import os
from transformers import pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "distilbert_model")
MODEL_DIR = os.path.abspath(MODEL_DIR)

classifier = pipeline("text-classification", model=MODEL_DIR)

LABEL_MAP = {
    "LABEL_0": "Fake",
    "LABEL_1": "Real",
    "Fake": "Fake",
    "Real": "Real",
}

def predict(text):
    result = classifier(text[:512])[0]

    raw_label = result["label"]
    label = LABEL_MAP.get(raw_label, raw_label)
    confidence = result["score"]

    return label, confidence