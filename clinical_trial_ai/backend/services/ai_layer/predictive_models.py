import asyncio
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from pathlib import Path


class PredictiveModels:
    """Advanced predictive models for clinical trial document analysis"""

    def __init__(self, model_cache_dir: str = "./model_cache"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Initialize NLP model
        self.nlp = self._load_spacy_model()
        
        # Initialize models
        self.document_classifier = None
        self.risk_assessor = None
        self.outcome_predictor = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Clinical trial specific patterns
        self.clinical_patterns = {
            'phase': r'phase\s+([ivx]+|\d+)',
            'enrollment': r'enrollment[:\s]*(\d+(?:,\d+)*)',
            'primary_endpoint': r'primary\s+endpoint[s]?[:\s]*([^.]+)',
            'adverse_events': r'adverse\s+event[s]?[:\s]*([^.]+)',
            'efficacy': r'efficacy[:\s]*([^.]+)',
            'safety': r'safety[:\s]*([^.]+)',
            'dosage': r'dosage[:\s]*([^.]+)',
            'inclusion_criteria': r'inclusion\s+criteria[:\s]*([^.]+)',
            'exclusion_criteria': r'exclusion\s+criteria[:\s]*([^.]+)'
        }
        
        # Risk indicators
        self.risk_indicators = [
            'severe', 'serious', 'adverse', 'toxicity', 'contraindication',
            'warning', 'caution', 'risk', 'side effect', 'complication',
            'mortality', 'death', 'fatal', 'life-threatening'
        ]
        
        # Success indicators
        self.success_indicators = [
            'efficacy', 'effective', 'successful', 'positive', 'beneficial',
            'improvement', 'response', 'remission', 'cure', 'healing',
            'statistically significant', 'p-value', 'confidence interval'
        ]

    def _load_spacy_model(self):
        """Load spaCy model for NLP processing"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            try:
                return spacy.load("en_core_web_lg")
            except OSError:
                print("Warning: No spaCy model found. Install with: python -m spacy download en_core_web_sm")
                return None

    async def predict(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Comprehensive prediction on document chunks"""
        results = []
        
        for chunk in chunks:
            chunk_id = chunk.get("id", f"chunk_{len(results)}")
            content = chunk.get("content", "")
            
            # Extract clinical information
            clinical_info = self._extract_clinical_info(content)
            
            # Predict document type
            doc_type = await self._predict_document_type(content)
            
            # Assess risk level
            risk_assessment = await self._assess_risk_level(content)
            
            # Predict trial outcome potential
            outcome_prediction = await self._predict_trial_outcome(content)
            
            # Extract key entities
            entities = self._extract_entities(content)
            
            # Calculate confidence based on content quality
            confidence = self._calculate_confidence(content, clinical_info)
            
            results.append({
                "chunk_id": chunk_id,
                "document_type": doc_type,
                "risk_level": risk_assessment["level"],
                "risk_score": risk_assessment["score"],
                "outcome_prediction": outcome_prediction["prediction"],
                "outcome_confidence": outcome_prediction["confidence"],
                "clinical_info": clinical_info,
                "entities": entities,
                "confidence": confidence,
                "content_length": len(content),
                "medical_terms_count": len(clinical_info.get("medical_terms", []))
            })
        
        return results

    def _extract_clinical_info(self, text: str) -> Dict[str, Any]:
        """Extract clinical trial specific information"""
        clinical_info = {
            "phase": None,
            "enrollment": None,
            "primary_endpoint": None,
            "adverse_events": None,
            "medical_terms": [],
            "dosages": [],
            "conditions": []
        }
        
        text_lower = text.lower()
        
        # Extract phase information
        phase_match = re.search(self.clinical_patterns['phase'], text_lower)
        if phase_match:
            clinical_info["phase"] = phase_match.group(1)
        
        # Extract enrollment numbers
        enrollment_match = re.search(self.clinical_patterns['enrollment'], text_lower)
        if enrollment_match:
            clinical_info["enrollment"] = enrollment_match.group(1)
        
        # Extract primary endpoint
        endpoint_match = re.search(self.clinical_patterns['primary_endpoint'], text_lower)
        if endpoint_match:
            clinical_info["primary_endpoint"] = endpoint_match.group(1).strip()
        
        # Extract adverse events
        ae_match = re.search(self.clinical_patterns['adverse_events'], text_lower)
        if ae_match:
            clinical_info["adverse_events"] = ae_match.group(1).strip()
        
        # Extract medical terms using patterns
        drug_pattern = r'\b[A-Z][a-z]+(?:in|ol|ide|ine|ate|mab|nib|zumab)\b'
        dosage_pattern = r'\d+(?:\.\d+)?\s*(?:mg|ml|g|l|units?|mcg|μg)\b'
        condition_pattern = r'\b(?:cancer|diabetes|hypertension|pneumonia|alzheimer|parkinson|depression|anxiety|asthma|copd|heart disease|stroke|seizure|epilepsy)\b'
        
        clinical_info["medical_terms"] = re.findall(drug_pattern, text)
        clinical_info["dosages"] = re.findall(dosage_pattern, text)
        clinical_info["conditions"] = re.findall(condition_pattern, text_lower)
        
        return clinical_info

    async def _predict_document_type(self, text: str) -> str:
        """Predict the type of clinical document"""
        text_lower = text.lower()
        
        # Simple rule-based classification
        if any(keyword in text_lower for keyword in ['protocol', 'study protocol', 'trial protocol']):
            return "protocol"
        elif any(keyword in text_lower for keyword in ['consent', 'informed consent', 'patient consent']):
            return "consent_form"
        elif any(keyword in text_lower for keyword in ['case report', 'adverse event', 'safety report']):
            return "safety_report"
        elif any(keyword in text_lower for keyword in ['results', 'outcome', 'efficacy', 'primary endpoint']):
            return "results"
        elif any(keyword in text_lower for keyword in ['enrollment', 'recruitment', 'participants']):
            return "enrollment"
        else:
            return "general_clinical"

    async def _assess_risk_level(self, text: str) -> Dict[str, Any]:
        """Assess risk level of the content"""
        text_lower = text.lower()
        
        # Count risk indicators
        risk_count = sum(1 for indicator in self.risk_indicators if indicator in text_lower)
        
        # Calculate risk score (0-1)
        risk_score = min(risk_count / 10.0, 1.0)  # Normalize to 0-1
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "level": risk_level,
            "score": risk_score,
            "indicators_found": risk_count
        }

    async def _predict_trial_outcome(self, text: str) -> Dict[str, Any]:
        """Predict potential trial outcome based on content"""
        text_lower = text.lower()
        
        # Count success indicators
        success_count = sum(1 for indicator in self.success_indicators if indicator in text_lower)
        
        # Look for statistical significance
        has_significance = any(term in text_lower for term in ['p < 0.05', 'p<0.05', 'statistically significant', 'significant difference'])
        
        # Calculate outcome confidence
        confidence = min((success_count + (2 if has_significance else 0)) / 15.0, 1.0)
        
        # Predict outcome
        if confidence >= 0.6:
            prediction = "positive"
        elif confidence >= 0.3:
            prediction = "neutral"
        else:
            prediction = "negative"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "has_statistical_significance": has_significance
        }

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        entities = {
            "drugs": [],
            "conditions": [],
            "organizations": [],
            "dates": [],
            "numbers": []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Skip person names for privacy
                    continue
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "CARDINAL":
                    entities["numbers"].append(ent.text)
                elif ent.label_ == "GPE":  # Geopolitical entity
                    entities["organizations"].append(ent.text)
        
        # Extract drugs and conditions using patterns
        drug_pattern = r'\b[A-Z][a-z]+(?:in|ol|ide|ine|ate|mab|nib|zumab)\b'
        condition_pattern = r'\b(?:cancer|diabetes|hypertension|pneumonia|alzheimer|parkinson|depression|anxiety|asthma|copd|heart disease|stroke|seizure|epilepsy|tumor|neoplasm|metastasis)\b'
        
        entities["drugs"] = list(set(re.findall(drug_pattern, text)))
        entities["conditions"] = list(set(re.findall(condition_pattern, text.lower())))
        
        return entities

    def _calculate_confidence(self, text: str, clinical_info: Dict[str, Any]) -> float:
        """Calculate confidence score based on content quality and clinical information"""
        confidence_factors = []
        
        # Length factor (longer text generally more informative)
        length_factor = min(len(text) / 1000.0, 1.0)
        confidence_factors.append(length_factor * 0.2)
        
        # Medical terms factor
        medical_terms_count = len(clinical_info.get("medical_terms", []))
        medical_factor = min(medical_terms_count / 10.0, 1.0)
        confidence_factors.append(medical_factor * 0.3)
        
        # Clinical information completeness
        clinical_fields = ["phase", "enrollment", "primary_endpoint", "adverse_events"]
        clinical_completeness = sum(1 for field in clinical_fields if clinical_info.get(field)) / len(clinical_fields)
        confidence_factors.append(clinical_completeness * 0.3)
        
        # Structure factor (presence of clinical trial keywords)
        clinical_keywords = ['trial', 'study', 'protocol', 'endpoint', 'efficacy', 'safety', 'adverse']
        keyword_count = sum(1 for keyword in clinical_keywords if keyword in text.lower())
        structure_factor = min(keyword_count / len(clinical_keywords), 1.0)
        confidence_factors.append(structure_factor * 0.2)
        
        return sum(confidence_factors)

    async def train_models(self, training_data: List[Dict[str, Any]]):
        """Train machine learning models on provided data"""
        if not training_data:
            return
        
        # Extract features and labels
        texts = [item.get("content", "") for item in training_data]
        labels = [item.get("label", "unknown") for item in training_data]
        
        # Vectorize texts
        X = self.tfidf_vectorizer.fit_transform(texts)
        
        # Train document classifier
        self.document_classifier = MultinomialNB()
        self.document_classifier.fit(X, labels)
        
        # Save models
        await self._save_models()

    async def _save_models(self):
        """Save trained models to disk"""
        if self.document_classifier:
            joblib.dump(self.document_classifier, self.model_cache_dir / "document_classifier.pkl")
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, self.model_cache_dir / "tfidf_vectorizer.pkl")

    async def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            classifier_path = self.model_cache_dir / "document_classifier.pkl"
            vectorizer_path = self.model_cache_dir / "tfidf_vectorizer.pkl"
            
            if classifier_path.exists():
                self.document_classifier = joblib.load(classifier_path)
            if vectorizer_path.exists():
                self.tfidf_vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            print(f"Warning: Could not load pre-trained models: {e}")

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about the predictive models"""
        return {
            "models_loaded": {
                "document_classifier": self.document_classifier is not None,
                "risk_assessor": self.risk_assessor is not None,
                "outcome_predictor": self.outcome_predictor is not None
            },
            "cache_directory": str(self.model_cache_dir),
            "clinical_patterns_count": len(self.clinical_patterns),
            "risk_indicators_count": len(self.risk_indicators),
            "success_indicators_count": len(self.success_indicators)
        }
