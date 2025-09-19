"""
Pipeline integrating cleaning, medical terminology extraction, and chunking.
Automatically downloads spaCy and NLTK models if missing.
"""

from typing import List, Dict
import spacy
import re
import nltk

from backend.services.text_processing.cleaning import ClinicalTextCleaner
from backend.services.text_processing.chunking import ClinicalTextChunker

# -----------------------------
# Auto-download NLTK resources
# -----------------------------
for resource, path in [("punkt_tab", "tokenizers/punkt_tab/english"),
                       ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# -----------------------------
# Clinical Text Processor
# -----------------------------
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

    # -----------------------------
    # SpaCy Model Loading
    # -----------------------------
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

    # -----------------------------
    # Medical Term Extraction
    # -----------------------------
    def extract_medical_terms(self, text: str) -> List[str]:
        terms: List[str] = []

        # NLP entities
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['CHEMICAL', 'DISEASE', 'GENE_OR_GENE_PRODUCT']:
                terms.append(ent.text.lower())

        # Regex patterns
        for pattern in self.MEDICAL_PATTERNS.values():
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend([m.lower() for m in matches])

        return list(set(terms))

    # -----------------------------
    # Full Text Processing
    # -----------------------------
    def process_text(self, text: str) -> Dict[str, List[str]]:
        cleaned_text = self.cleaner.clean_text(text)
        medical_terms = self.extract_medical_terms(cleaned_text)
        chunks = self.chunker.chunk_text(cleaned_text)

        return {
            "cleaned_text": cleaned_text,
            "medical_terms": medical_terms,
            "chunks": chunks
        }


# -----------------------------
# Mock Preprocessing Test
# -----------------------------
import difflib


def mock_preprocessing_test():
    processor = ClinicalTextProcessor(chunk_size=50, chunk_overlap=10)

    sample_text = """
    Pt was admitted w/ pneumonia and prescribed 500mg amoxicillin. The patient has a history of diabetes and hypertension. MRI and CT scan showed no abnormalities. 
    Vitals were stable, but the patient reported mild chest pain and shortness of breath. Blood tests revealed elevated white blood cell count and mild anemia. 
    The patient underwent an echocardiogram, which was unremarkable. The treatment plan included physiotherapy, dietary recommendations, and follow-up lab tests. 
    The patient was educated about medication adherence, monitoring blood sugar, and reporting any new symptoms. 
    Follow-up visits scheduled every two weeks for the next three months. Additional notes: patient has a history of asthma, seasonal allergies, and prior surgery for appendicitis. 
    Prescribed medications: insulin, amlodipine, atorvastatin, and albuterol inhaler as needed. Patient advised to avoid smoking and strenuous activity. 
    Further imaging: chest X-ray, abdominal ultrasound, and MRI of the lumbar spine. Blood pressure monitored daily; ECG normal. 
    Patient reported mild dizziness on standing; orthostatic vitals recorded. Nutrition consultation completed, and low-sodium diet recommended. 
    Patient discharged with home health care and daily nursing visits. Education materials provided regarding hypertension management and diabetes care. 
    Patient follow-up labs: CBC, CMP, HbA1c, lipid panel. Patient instructed to report any fever, chest pain, or new shortness of breath immediately. 
    """

    # Run cleaning
    cleaned_text = processor.cleaner.clean_text(sample_text)

    # Show differences
    print("=== Differences (Original vs Cleaned) ===")
    diff = difflib.unified_diff(
        sample_text.splitlines(),
        cleaned_text.splitlines(),
        fromfile='Original',
        tofile='Cleaned',
        lineterm=''
    )
    print('\n'.join(diff))
    #
    # # Run full pipeline
    # result = processor.process_text(sample_text)
    #
    # print("\n=== Medical Terms ===")
    # print(result["medical_terms"])
    # print("\n=== Chunks ===")
    # for i, chunk in enumerate(result["chunks"]):
    #     print(f"Chunk {i+1}: {chunk}\n")


if __name__ == "__main__":
    mock_preprocessing_test()

