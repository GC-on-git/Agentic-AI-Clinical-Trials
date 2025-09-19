# backend/services/text_processing/cleaner.py
"""
Advanced text cleaning and normalization for clinical documents.
Includes lowercasing, stopword removal, lemmatization, negation handling, and medical abbreviation normalization.
"""

import re
from typing import List
import nltk
from nltk.corpus import stopwords
import spacy


for resource, path in [("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# Load English stopwords
STOP_WORDS = set(stopwords.words("english"))

# Load spaCy English model for lemmatization
try:
    nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


class ClinicalTextCleaner:
    """Cleans and normalizes clinical text."""
    def __init__(self):
        self.nlp = self._load_spacy_model()

    def _load_spacy_model(self):
        """Load spaCy model, download only if missing."""
        for model_name in ["en_core_web_lg", "en_core_web_sm"]:
            try:
                # First try to load the model
                return spacy.load(model_name)
            except OSError:
                # Model not found, download and then load
                try:
                    spacy.cli.download(model_name)
                    return spacy.load(model_name)
                except Exception:
                    continue
        raise RuntimeError("Failed to load any spaCy model.")

    MEDICAL_ABBREV = {
        'w/': 'with',
        'w/o': 'without',
        'pt': 'patient',
        'pts': 'patients',
        'dx': 'diagnosis',
        'tx': 'treatment',
        'rx': 'prescription',
        'h/o': 'history of',
        's/p': 'status post',
    }

    NEGATIONS = {
        "no": "not",
        "never": "not",
        "none": "not",
        "without": "not",
    }

    @staticmethod
    def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """Clean and normalize input clinical text."""

        # 1️⃣ Lowercase text
        text = text.lower()

        # 2️⃣ Fix common OCR errors
        text = re.sub(r'\bl\b', 'i', text)
        text = re.sub(r'\bO\b(?=\d)', '0', text)

        # 3️⃣ Expand medical abbreviations
        for abbrev, full_form in ClinicalTextCleaner.MEDICAL_ABBREV.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, text, flags=re.IGNORECASE)

        # 4️⃣ Expand negations
        for neg, replacement in ClinicalTextCleaner.NEGATIONS.items():
            text = re.sub(rf'\b{neg}\b', replacement, text)

        # 5️⃣ Remove extra whitespace and punctuation (keep basic medical characters)
        text = re.sub(r'[^a-z0-9\s\.\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 6️⃣ Tokenize, remove stopwords, and lemmatize
        doc = nlp(text)
        tokens = []
        for token in doc:
            if remove_stopwords and token.text in STOP_WORDS:
                continue
            if lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        cleaned_text = ' '.join(tokens)
        return cleaned_text
