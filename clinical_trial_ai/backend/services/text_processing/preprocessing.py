# backend/services/text_processing/preprocessing.py
"""
Pipeline integrating cleaning, medical terminology extraction, and chunking.
"""

from typing import List, Dict, Any
import spacy
import re
import nltk

from clinical_trial_ai.backend.services.text_processing.cleaning import ClinicalTextCleaner
from clinical_trial_ai.backend.services.text_processing.chunking import ClinicalTextChunker

# Auto-download NLTK resources
for resource, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

class ClinicalTextProcessor:
    """Processes clinical documents in a pipeline."""

    MEDICAL_PATTERNS = {
        'drug_names': r'\b[A-Z][a-z]+(?:in|ol|ide|ine|ate)\b',
        'dosages': r'\d+(?:\.\d+)?\s*(?:mg|ml|g|l|units?)\b',
        'conditions': r'\b(?:cancer|diabetes|hypertension|pneumonia)\b',
        'procedures': r'\b(?:surgery|biopsy|CT scan|MRI|X-ray)\b'
    }

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.cleaner = ClinicalTextCleaner()
        self.chunker = ClinicalTextChunker(chunk_size, chunk_overlap)
        self.nlp = self._load_spacy_model()

    def _load_spacy_model(self):
        for model_name in ["en_core_web_lg", "en_core_web_sm"]:
            try:
                return spacy.load(model_name)
            except OSError:
                try:
                    spacy.cli.download(model_name)
                    return spacy.load(model_name)
                except Exception:
                    continue
        raise RuntimeError("Failed to load any spaCy model.")

    def extract_medical_terms(self, text: str) -> List[str]:
        terms: List[str] = []

        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['CHEMICAL', 'DISEASE', 'GENE_OR_GENE_PRODUCT']:
                terms.append(ent.text.lower())

        for pattern in self.MEDICAL_PATTERNS.values():
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend([m.lower() for m in matches])

        return list(set(terms))

    def process_text(self, text: str, document_id: str) -> dict[str, Any]:
        """
        Full processing pipeline:
        - Cleaning
        - Medical terminology extraction
        - Chunking into pipeline-ready dicts
        """
        cleaned_text = self.cleaner.clean_text(text)
        medical_terms = self.extract_medical_terms(cleaned_text)
        chunks = self.chunker.chunk_text(cleaned_text, document_id=document_id)

        return {
            "cleaned_text": cleaned_text,
            "medical_terms": medical_terms,
            "chunks": chunks  # chunks are now dicts with content + metadata
        }
