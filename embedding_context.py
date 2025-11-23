"""
Embedding-based context verification for health misinformation detection.
Uses sentence transformers to detect semantic context and refutations.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    print("Falling back to rule-based context detection only.")


class EmbeddingContextVerifier:
    """
    Uses sentence embeddings to verify context and detect refutations.
    Falls back to rule-based if embeddings are not available.
    """
    
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE
        self.model: Optional[SentenceTransformer] = None
        
        if self.use_embeddings:
            try:
                # Use a lightweight, fast model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._initialize_reference_embeddings()
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                print("Falling back to rule-based context detection.")
                self.use_embeddings = False
    
    def _initialize_reference_embeddings(self):
        """Initialize reference embeddings for refutation detection."""
        if not self.model:
            return
        
        # Reference patterns for refutation detection
        self.refutation_refs = [
            "this is false",
            "this is not true",
            "this is misinformation",
            "this claim is false",
            "this has been debunked",
            "this is a myth",
            "this is wrong",
            "this is incorrect",
            "this is dangerous",
            "this is harmful",
            "do not do this",
            "this is not recommended",
        ]
        
        # Reference patterns for legitimate health education
        self.legitimate_refs = [
            "talk to your doctor",
            "consult your physician",
            "seek medical advice",
            "according to medical research",
            "studies show",
            "clinical trial",
            "peer-reviewed research",
            "evidence-based",
        ]
        
        # Generate embeddings for reference patterns
        self.refutation_embeddings = self.model.encode(self.refutation_refs)
        self.legitimate_embeddings = self.model.encode(self.legitimate_refs)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def detect_refutation_context(self, text: str) -> Tuple[bool, float]:
        """
        Detect if text contains refutation context.
        Returns (is_refutation, confidence_score)
        """
        if not self.use_embeddings or not self.model:
            # Fallback to simple rule-based check
            import re
            refutation_keywords = ['false', 'not true', 'misinformation', 'debunked', 'myth', 'wrong']
            text_lower = text.lower()
            for keyword in refutation_keywords:
                if keyword in text_lower:
                    return (True, 0.5)
            return (False, 0.0)
        
        try:
            # Encode the input text
            text_embedding = self.model.encode([text])[0]
            
            # Compare with refutation references
            max_refutation_sim = 0.0
            for ref_emb in self.refutation_embeddings:
                sim = self.cosine_similarity(text_embedding, ref_emb)
                max_refutation_sim = max(max_refutation_sim, sim)
            
            # Compare with legitimate health education references
            max_legitimate_sim = 0.0
            for legit_emb in self.legitimate_embeddings:
                sim = self.cosine_similarity(text_embedding, legit_emb)
                max_legitimate_sim = max(max_legitimate_sim, sim)
            
            # If refutation similarity is high, it's likely a refutation
            if max_refutation_sim > 0.5:
                return (True, max_refutation_sim)
            
            # If legitimate similarity is high and refutation is low, it's legitimate
            if max_legitimate_sim > 0.6 and max_refutation_sim < 0.4:
                return (True, max_legitimate_sim)  # Treat as safe context
            
            return (False, 0.0)
        
        except Exception as e:
            print(f"Error in embedding detection: {e}")
            return (False, 0.0)
    
    def verify_source_usage(self, text: str, source_mention: str) -> Tuple[bool, float]:
        """
        Verify if a source is being used correctly (supporting) vs misused.
        Returns (is_legitimate_use, confidence)
        """
        if not self.use_embeddings or not self.model:
            # Fallback: simple pattern matching
            import re
            legitimate_patterns = [
                rf"{re.escape(source_mention)}\s+(says|states|reports|finds|shows)",
                rf"according\s+to\s+{re.escape(source_mention)}",
            ]
            if any(re.search(p, text, re.IGNORECASE) for p in legitimate_patterns):
                return (True, 0.6)
            return (False, 0.0)
        
        try:
            # Extract sentences containing the source
            import re
            sentences = re.split(r'[.!?]+', text)
            source_sentences = [s for s in sentences if source_mention.lower() in s.lower()]
            
            if not source_sentences:
                return (False, 0.0)
            
            # Reference patterns for legitimate vs misuse
            legitimate_patterns = [
                f"{source_mention} says",
                f"{source_mention} reports",
                f"according to {source_mention}",
                f"{source_mention} study shows",
            ]
            
            misuse_patterns = [
                f"{source_mention} says but",
                f"despite {source_mention}",
                f"{source_mention} is wrong",
            ]
            
            # Encode source sentences and compare
            source_text = " ".join(source_sentences)
            source_embedding = self.model.encode([source_text])[0]
            
            legit_embeddings = self.model.encode(legitimate_patterns)
            misuse_embeddings = self.model.encode(misuse_patterns)
            
            # Find max similarity
            max_legit_sim = max(
                self.cosine_similarity(source_embedding, emb) 
                for emb in legit_embeddings
            )
            max_misuse_sim = max(
                self.cosine_similarity(source_embedding, emb) 
                for emb in misuse_embeddings
            )
            
            if max_legit_sim > 0.6 and max_misuse_sim < 0.4:
                return (True, max_legit_sim)
            elif max_misuse_sim > 0.5:
                return (False, max_misuse_sim)
            
            return (False, 0.0)
        
        except Exception as e:
            print(f"Error in source verification: {e}")
            return (False, 0.0)
    
    def get_context_adjustment(self, text: str) -> float:
        """
        Get overall context adjustment factor based on embeddings.
        Returns negative value to reduce scores, positive to increase.
        """
        is_refutation, confidence = self.detect_refutation_context(text)
        
        if is_refutation:
            # Strong reduction for refutations
            return -0.6 * confidence
        
        return 0.0

