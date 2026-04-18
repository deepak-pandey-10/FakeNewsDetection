import os
from transformers import pipeline

# Pulling the model directly from the Hugging Face Hub!
HF_MODEL_ID = "deepak002p/FakeNews-DistilBERT"

classifier = pipeline("text-classification", model=HF_MODEL_ID)

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