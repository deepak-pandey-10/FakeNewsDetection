import re

def extract_claims(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if len(s) > 20][:5]