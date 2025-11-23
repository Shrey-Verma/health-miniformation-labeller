from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
import re
from pathlib import Path
from urllib.parse import urlparse
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Fallback sentence splitting
    def sent_tokenize(text: str) -> List[str]:
        # Simple sentence splitting as fallback
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

# Try to import embedding context verifier
try:
    from embedding_context import EmbeddingContextVerifier
    EMBEDDING_VERIFIER_AVAILABLE = True
except ImportError:
    EMBEDDING_VERIFIER_AVAILABLE = False
    EmbeddingContextVerifier = None

LABEL_UNVERIFIED_CURE = "potential-unverified-cure"
LABEL_UNSAFE_MED_ADVICE = "potential-unsafe-medication-advice"
LABEL_RISKY_FASTING_DETOX = "risky-fasting-detox-content"
LABEL_UNVERIFIED_SUPPLEMENT = "unverified-supplement-claims"
LABEL_UNSAFE_DEVICE_USE = "unsafe-device-usage"

UNVERIFIED_CURE_PATTERNS = [
    r"\b(cures?|reverse[sd]?|eliminates?)\b.*\b(cancer|diabetes|autism|heart\s+disease)\b",
    r"\bmiracle (cure|treatment)\b",
    r"\bno side effects\b.*\b(any|all)\b.*\b(disease|condition)\b",
]

UNSAFE_MED_PATTERNS = [
    r"\b(stop|quit|throw away|ditch)\b.*\b(insulin|statins?|antidepressants?|blood pressure meds?|bp meds?)\b",
    r"\b(double|triple|cut|halve)\s+(your|the)\s+(dose|dosage)\b",
    r"\b(no need|don['']t need)\b.*\b(prescription|doctor)\b",
    r"\b(double|triple|cut|halve)\s+(your|a)\s+(dose|dosage)\b.*\b(supplement|medication|meds?)\b",
]

RISKY_FASTING_DETOX_PATTERNS = [
    r"\b(dry fast|water fast)\b.*\b(48|72|96|120)\s*(h|hours?)\b",
    r"\bdetox (tea|cleanse|flush)\b.*\b(weight|fat|toxins?)\b",
    r"\bcolon cleanse\b.*\b(lose|drop)\s+\d+\s*(kg|kgs|pounds|lbs)\b",
    r"\b(detox|cleanse)\b.*\b(tea|drink)\b",  # More general detox patterns
    r"\bdrink\b.*\b(detox|cleanse)\b.*\b(tea|instead)\b",  # "drink this detox tea instead"
]

UNVERIFIED_SUPPLEMENT_PATTERNS = [
    r"\bNMN|NAD\+|peptides?|SARMs?|nootropics?\b.*\b(cures?|fix(es|ed)|revers(es|e))\b",
    r"\bjust take\b.*\b(supplements?|herbs?|pills?)\b.*\b(no need|instead)\b.*\b(doctor|medication|treatment)\b",
    r"\b(supplements?|herbs?|pills?)\b.*\b(instead|instead of)\b.*\b(medications?|meds?|doctor|treatment)\b",
    r"\b(double|increase)\s+(your|the)\s+(dose|dosage)\b.*\b(supplement|herb|pill)\b.*\b(instead of|instead)\b",
]

UNSAFE_DEVICE_PATTERNS = [
    r"\bnebuliz(e|er|ing)\b.*\b(essential oils?|hydrogen peroxide|vinegar)\b",
    r"\b(essential oils?|hydrogen peroxide|vinegar)\b.*\bnebuliz(e|er|ing)\b",  # Reverse order
    r"\bCGM\b.*\b(hack|game|cheat)\b.*\b(diabetes|insulin)\b",
    r"\bSpO2\b.*\b(no need\b.*\bdoctor|ignore\b.*\bdoctor)\b",
    r"\buse\b.*\b(essential oils?)\b.*\b(nebuliz|inhal)\b",  # More flexible pattern
]

ALLOW_CONTEXT_PATTERNS = [
    r"\b(CDC|WHO|NIH|NHS|EMA|Mayo Clinic|Cochrane)\b",
    r"\b(systematic review|meta[- ]analysis|randomized (controlled )?trial)\b",
    r"\b(talk to|speak with|check with)\s+(your|a)\s+doctor\b",
    r"\bmyth\b.*\bdebunk(ed|ing)?\b",
    r"\b(fact[- ]?check(ed|ing)?|fact\s+checked)\b",
]

REFUTATION_PATTERNS = [
    r"\b(is|are)\s+not\s+true\b",
    r"\bthis is misinformation\b",
    r"\bthis claim is false\b",
    r"\bdebunk(ing|ed)\b",
    r"\b(is|are)\s+false\b",
    r"\b(is|are)\s+wrong\b",
    r"\bhas been debunked\b",
    r"\bproven false\b",
    r"\bnot supported by evidence\b",
]

# Negation indicators that suggest the harmful content is being refuted
NEGATION_INDICATORS = [
    r"\b(but|however|although|though)\b",
    r"\b(not|don't|doesn't|didn't|won't|wouldn't|shouldn't)\b",
    r"\b(never|no one should|you should not|avoid)\b",
    r"\b(this is|that is|which is)\s+(false|wrong|incorrect|dangerous|harmful|not true)\b",
    r"\b(contrary to|despite|in spite of)\b",
]

# Quote indicators - suggests content is being quoted rather than promoted
QUOTE_INDICATORS = [
    r'["\']',  # Quotation marks
    r"\b(some|people|they|others)\s+(say|claim|argue|believe|think)\b",
    r"\b(allegedly|supposedly|reportedly)\b",
    r"\b(according to|as|quote|quoted)\b",
]

CERTAINTY_WORDS = [
    r"\bnever\b", r"\balways\b", r"\b100%\b", r"\bguaranteed?\b", r"\bfor sure\b"
]

IMPERATIVE_HEALTH_PATTERNS = [
    r"\b(stop|quit|ditch|throw away|skip|avoid|refuse)\b.*\b(shots?|vaccines?|meds?|medications?|insulin)\b",
    r"\bjust\b.*\b(take|use|drink|eat|do)\b.*\b(instead)\b",
]

URL_REGEX = re.compile(r"https?://\S+")


def load_domain_list(path: Path) -> Set[str]:
    domains: Set[str] = set()
    if not path.exists():
        return domains
    with path.open() as f:
        for line in f:
            dom = line.strip().lower()
            if dom and not dom.startswith("#"):
                domains.add(dom)
    return domains


def extract_domains(text: str):
    urls = URL_REGEX.findall(text)
    domains = []
    for u in urls:
        try:
            netloc = urlparse(u).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            domains.append(netloc)
        except Exception:
            continue
    return domains


def any_match(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def count_matches(patterns, text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK if available, otherwise simple regex."""
    if NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except:
            pass
    # Fallback: simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def check_negation_context(sentence: str, pattern_match: bool) -> bool:
    """
    Check if a sentence containing a harmful pattern is negated or refuted.
    Returns True if the pattern should be ignored due to negation.
    """
    if not pattern_match:
        return False
    
    sentence_lower = sentence.lower()
    

    source_orgs = ['cdc', 'who', 'nih', 'nhs', 'ema', 'mayo clinic', 'cochrane']
    source_misuse_patterns = [
        rf"\b({'|'.join(re.escape(org) for org in source_orgs)})\s+is\s+(wrong|false|lying|incorrect)\b",
        rf"\b({'|'.join(re.escape(org) for org in source_orgs)})\s+(says?|claims?)\s+.*\b(but|however|actually|really)\b",
        rf"\bdespite\s+({'|'.join(re.escape(org) for org in source_orgs)})\b",
    ]
    
    # If sentence contains source misuse, don't treat as negation
    # This prevents "CDC is wrong, stop taking insulin" from being incorrectly negated
    has_source_misuse = any_match(source_misuse_patterns, sentence)
    if has_source_misuse:
        return False  # Source misuse is not negation of harmful advice
    
    # Check for explicit refutation patterns (but only if not source misuse)
    if any_match(REFUTATION_PATTERNS, sentence):
        return True
    

    harmful_action_negations = [
        r"\b(don't|do not|never|shouldn't|wouldn't)\s+(stop|quit|ditch|throw away|skip)\b",
        r"\b(don't|do not|never|shouldn't|wouldn't)\s+(take|use|drink|eat)\b.*\b(instead|instead of)\b",
        r"\b(should|must|always)\s+(not|never)\s+(stop|quit|skip|avoid)\b",
    ]
    
    # If we see negation of harmful actions, it's a refutation
    if any_match(harmful_action_negations, sentence):
        return True
    
    # Check for negation indicators near the pattern, but be more precise
    # "Don't trust doctors" is harmful, not a refutation
    # "Don't stop taking insulin" is a refutation
    # "CDC is wrong, stop taking insulin" is NOT a refutation (source misuse)
    negation_words = ['not', "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", 
                      'never', 'false', 'wrong', 'incorrect', 'dangerous', 'harmful']
    
    # Look for negation within context of harmful keywords
    harmful_keywords = ['cure', 'stop', 'quit', 'fast', 'detox', 'supplement', 'nebulizer']
    for keyword in harmful_keywords:
        if keyword in sentence_lower:
            keyword_pos = sentence_lower.find(keyword)
            # Check 30 chars before for negation
            context_before = sentence_lower[max(0, keyword_pos - 30):keyword_pos]
            
            # Check if negation is before the keyword (suggesting refutation)
            # But exclude cases like "don't trust" which are themselves harmful
            if any(neg in context_before for neg in ['not', "don't", "doesn't"]):
                # Check if it's negating the action itself
                if re.search(rf"\b(don't|do not|not)\s+{keyword}", sentence_lower):
                    return True
            
         
            if not has_source_misuse:
                refutation_words = ['false', 'wrong', 'incorrect', 'myth', 'debunk']
                for ref_word in refutation_words:
                    if ref_word in context_before:
                        # Check if refutation word is in a pattern that clearly refers to the harmful action
                        # Patterns like: "this [action] is wrong", "that [action] is false", "[action] is not true"
                        refutation_patterns = [
                            rf"\b(this|that|the)\s+.*\b{re.escape(keyword)}.*\b(is|are)\s+{ref_word}\b",
                            rf"\b{re.escape(keyword)}.*\b(is|are)\s+(not\s+)?(true|correct|valid)\b",
                            rf"\b(this|that|the)\s+.*\b{re.escape(keyword)}.*\b(has\s+been\s+)?(debunk|myth)\b",
                            rf"\b(this|that|the)\s+.*\b(is|are)\s+{ref_word}.*\b{re.escape(keyword)}\b",  # "this is wrong about [action]"
                        ]
                        if any_match(refutation_patterns, sentence_lower):
                            return True
    
    return False


def check_quoted_content(text: str, sentence: str) -> bool:
    """
    Check if harmful content appears to be quoted rather than promoted.
    Returns True if content is likely quoted (should reduce score).
    """
    sentence_lower = sentence.lower()
    
    # Check if sentence is explicitly in quotes (balanced quotes)
    if '"' in sentence:
        quote_count = sentence.count('"')
        if quote_count >= 2:  # Balanced quotes suggest quoted content
            return True
    
    # Check for explicit quote/reporting language (more specific patterns)
    explicit_quote_patterns = [
        r'\b(some|people|they|others|many|critics)\s+(say|claim|argue|believe|think|suggest|allege)\s+["\']',
        r'["\'].*\b(say|claim|argue|believe|think|suggest|allege)\b',
        r'\b(according to|as|quote|quoted|reportedly|allegedly)\b.*["\']',
    ]
    
    if any_match(explicit_quote_patterns, sentence):
        return True
    
    # Check for reporting verbs with proper context (not just "don't trust")
    reporting_verbs = ['say', 'claim', 'argue', 'believe', 'think', 'suggest', 'allege']
    for verb in reporting_verbs:
        # More specific: "some say X" or "people claim X" (not "don't trust")
        if re.search(rf'\b(some|people|they|others|many|critics)\s+{verb}\b', sentence_lower):
            # Make sure it's not a direct command like "don't trust"
            if not re.search(rf'\b(don\'t|do not|never)\s+{verb}\b', sentence_lower):
                return True
    
    return False


def verify_source_citation(text: str, source_orgs: List[str]) -> Tuple[bool, float, bool]:
    """
    Verify if credible sources are being cited correctly (supporting) vs misused.
    Returns (has_adjustment, adjustment_value, has_any_misuse)
    - has_adjustment: True if there's any adjustment to apply (positive or negative)
    - adjustment_value: Positive value = reduction (decrease score), Negative value = penalty (increase score)
    - has_any_misuse: True if ANY source is being misused, regardless of other legitimate citations
    """
    text_lower = text.lower()
    adjustment = 0.0
    has_any_misuse = False
    sentences = split_into_sentences(text)
    
    for org in source_orgs:
        org_lower = org.lower()
        if org_lower not in text_lower:
            continue
        
        # Look for citation patterns that suggest legitimate use
        legitimate_patterns = [
            rf"{re.escape(org_lower)}\s+(says?|states?|reports?|finds?|shows?|confirms?|recommends?)",
            rf"according\s+to\s+{re.escape(org_lower)}",
            rf"{re.escape(org_lower)}\s+(study|research|trial|analysis|review)",
            rf"({re.escape(org_lower)}\s+)?(randomized|systematic|meta[- ]?analysis)",
        ]
        

        general_misuse_patterns = [
            rf"despite\s+{re.escape(org_lower)}",
            rf"{re.escape(org_lower)}\s+is\s+(wrong|false|lying)",
        ]
        

        sentence_constrained_misuse_pattern = rf"{re.escape(org_lower)}\s+(says?|claims?)\s+.*\b(but|however|actually|really)\b"
        
        # Check for misuse FIRST (takes priority over legitimate citation)
        # Bug 1 Fix: Check sentence-constrained pattern within individual sentences
        org_has_misuse = False
        
        # Check general misuse patterns across entire text
        if any_match(general_misuse_patterns, text):
            org_has_misuse = True
        
        # Check sentence-constrained pattern within each sentence containing the org
        if not org_has_misuse:
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if org_lower in sentence_lower:
                    # Check if this sentence contains the sentence-constrained misuse pattern
                    match = re.search(sentence_constrained_misuse_pattern, sentence_lower, flags=re.IGNORECASE)
                    if match:
                        # Bug 1 Fix: Check if "but/however" is followed by another source name
                        # If it is, the contradiction is about that other source, not the current one
                        match_end = match.end()
                        text_after_match = sentence_lower[match_end:]
                        
                        # List of other source names (excluding current org)
                        other_sources = [s.lower() for s in source_orgs if s.lower() != org_lower]
                        
                        # Check if another source name appears shortly after the "but" (within 50 chars)
                        # Pattern: "but WHO says" or "however CDC says" etc.
                        other_source_pattern = r'\b(' + '|'.join(re.escape(s) for s in other_sources) + r')\s+(says?|claims?|states?)'
                        if re.search(other_source_pattern, text_after_match[:50], flags=re.IGNORECASE):
                            # "but" introduces another source - not misuse of current source
                            continue
                        
                        # If no other source follows, it's likely misuse of current source
                        org_has_misuse = True
                        break
        
        has_legitimate = any_match(legitimate_patterns, text)
        
        if org_has_misuse:

            adjustment -= 0.3
            has_any_misuse = True  # Bug 2 Fix: Track misuse separately

        elif has_legitimate:
            # Check if it's being used to support vs refute
            for sentence in sentences:
                if org_lower in sentence.lower():
                    # If sentence contains refutation language, might be misused
                    if any_match(REFUTATION_PATTERNS, sentence):
                        adjustment += 0.2  # Partial reduction (positive = decrease score)
                    else:
                        adjustment += 0.5  # Full reduction for legitimate citation (positive = decrease score)
                    break
    
    return adjustment != 0.0, adjustment, has_any_misuse


def analyze_context_window(text: str, sentence_idx: int, sentences: List[str]) -> float:
    """
    Analyze surrounding sentences for context that might affect scoring.
    Returns a context adjustment factor (negative = reduce score, positive = increase).
    """
    context_score = 0.0
    
    # Get context window (current sentence + 2 before/after)
    start_idx = max(0, sentence_idx - 2)
    end_idx = min(len(sentences), sentence_idx + 3)
    context_sentences = sentences[start_idx:end_idx]
    context_text = " ".join(context_sentences)
    
    # Check for refutation in context
    if any_match(REFUTATION_PATTERNS, context_text):
        context_score -= 0.5
    
    # Check for safety advice in context
    safety_patterns = [
        r"\b(talk to|speak with|check with|consult)\s+(your|a)\s+doctor\b",
        r"\b(seek|get)\s+(medical|professional)\s+(advice|help|care)\b",
        r"\b(always|make sure to)\s+(consult|talk to)\s+(your|a)\s+doctor\b",
    ]
    if any_match(safety_patterns, context_text):
        context_score -= 0.4
    
    # Check for credible sources in context
    if any_match(ALLOW_CONTEXT_PATTERNS, context_text):
        context_score -= 0.3
    
    return context_score


class HealthPolicyScorer:
    def __init__(self, domain_dir: Path | None = None, use_embeddings: bool = True):
        domain_dir = domain_dir or Path("domain_lists")
        self.allow_domains = load_domain_list(domain_dir / "allow_domains.csv")
        self.risk_domains = load_domain_list(domain_dir / "risk_domains.csv")
        
        # Initialize embedding-based context verifier if available
        self.embedding_verifier = None
        if use_embeddings and EMBEDDING_VERIFIER_AVAILABLE and EmbeddingContextVerifier:
            try:
                self.embedding_verifier = EmbeddingContextVerifier(use_embeddings=True)
            except Exception as e:
                print(f"Warning: Could not initialize embedding verifier: {e}")
                self.embedding_verifier = None

    def score_text(self, text: str) -> Dict[str, float]:
        t = text.strip()
        scores: Dict[str, float] = {
            LABEL_UNVERIFIED_CURE: 0.0,
            LABEL_UNSAFE_MED_ADVICE: 0.0,
            LABEL_RISKY_FASTING_DETOX: 0.0,
            LABEL_UNVERIFIED_SUPPLEMENT: 0.0,
            LABEL_UNSAFE_DEVICE_USE: 0.0,
        }

        # Split into sentences for context-aware analysis
        sentences = split_into_sentences(t)
        
        # Track which patterns matched and in which sentences
        pattern_matches = {
            LABEL_UNVERIFIED_CURE: [],
            LABEL_UNSAFE_MED_ADVICE: [],
            LABEL_RISKY_FASTING_DETOX: [],
            LABEL_UNVERIFIED_SUPPLEMENT: [],
            LABEL_UNSAFE_DEVICE_USE: [],
        }

        # Check each sentence individually for patterns
        for sent_idx, sentence in enumerate(sentences):
            # Check each category
            if any_match(UNVERIFIED_CURE_PATTERNS, sentence):
                if not check_negation_context(sentence, True):
                    pattern_matches[LABEL_UNVERIFIED_CURE].append(sent_idx)
            
            if any_match(UNSAFE_MED_PATTERNS, sentence):
                if not check_negation_context(sentence, True):
                    pattern_matches[LABEL_UNSAFE_MED_ADVICE].append(sent_idx)
            
            if any_match(RISKY_FASTING_DETOX_PATTERNS, sentence):
                if not check_negation_context(sentence, True):
                    pattern_matches[LABEL_RISKY_FASTING_DETOX].append(sent_idx)
            
            if any_match(UNVERIFIED_SUPPLEMENT_PATTERNS, sentence):
                if not check_negation_context(sentence, True):
                    pattern_matches[LABEL_UNVERIFIED_SUPPLEMENT].append(sent_idx)
            
            if any_match(UNSAFE_DEVICE_PATTERNS, sentence):
                if not check_negation_context(sentence, True):
                    pattern_matches[LABEL_UNSAFE_DEVICE_USE].append(sent_idx)

        # Apply base scores for non-negated matches
        for label, matched_sentences in pattern_matches.items():
            if matched_sentences:
                base_score = 1.0
                # Check if content is quoted (reduce score)
                quoted_reduction = 0.0
                context_adjustments = []
                for sent_idx in matched_sentences:
                    if check_quoted_content(t, sentences[sent_idx]):
                        quoted_reduction += 0.3
                    # Analyze context window
                    context_adj = analyze_context_window(t, sent_idx, sentences)
                    context_adjustments.append(context_adj)
                
                # Use the most negative context adjustment (most conservative)
                min_context_adj = min(context_adjustments) if context_adjustments else 0.0
                quoted_reduction = min(quoted_reduction, 0.4)  # Cap quoted reduction
                
                scores[label] = base_score + min_context_adj - quoted_reduction

        # Domain-based adjustments
        domains = extract_domains(t)
        if domains:
            if any(d in self.risk_domains for d in domains):
                for k in scores:
                    scores[k] += 0.3
            if any(d in self.allow_domains for d in domains):
                for k in scores:
                    scores[k] -= 0.5

        # Source citation verification (more sophisticated)
        source_orgs = ['CDC', 'WHO', 'NIH', 'NHS', 'EMA', 'Mayo Clinic', 'Cochrane']
        has_citation_adjustment, citation_adjustment, source_misuse_detected = verify_source_citation(t, source_orgs)
        
        if has_citation_adjustment:
            # citation_adjustment is positive for legitimate citations (reduce score)
            # citation_adjustment is negative for misuse (increase score)
            for k in scores:
                scores[k] -= citation_adjustment  # Subtract: positive reduces, negative increases
        
        # Embedding-based context verification (if available)
        if self.embedding_verifier:
            embedding_adjustment = self.embedding_verifier.get_context_adjustment(t)
            for k in scores:
                scores[k] += embedding_adjustment
            
            # Enhanced source verification using embeddings
            for org in source_orgs:
                if org.lower() in t.lower():
                    is_legit, confidence = self.embedding_verifier.verify_source_usage(t, org)
                    if is_legit:
                        for k in scores:
                            scores[k] -= 0.3 * confidence
                    elif confidence > 0.5:  # High confidence it's misuse
                        source_misuse_detected = True
                        for k in scores:
                            scores[k] += 0.2 * confidence

        # Legacy context patterns (for backward compatibility and additional coverage)
        # Skip this reduction if source misuse was detected (don't reduce score for misused sources)
        if not source_misuse_detected:
            if any_match(ALLOW_CONTEXT_PATTERNS, t) or any_match(REFUTATION_PATTERNS, t):
                for k in scores:
                    scores[k] -= 0.5  # Reduced from 0.7 since we have better detection now

        # Stance amplification
        certainty_hits = count_matches(CERTAINTY_WORDS, t)
        imperative_hits = count_matches(IMPERATIVE_HEALTH_PATTERNS, t)
        stance_boost = 0.2 * certainty_hits + 0.3 * imperative_hits
        if stance_boost:
            for k in scores:
                if scores[k] > 0:
                    scores[k] += stance_boost

        # Ensure scores don't go negative
        for k in scores:
            scores[k] = max(0.0, scores[k])

        return scores

    def labels_for_text(self, text: str, mode: str = "default") -> List[str]:
        scores = self.score_text(text)
        if mode == "conservative":
            thresh = 1.2
        elif mode == "recall":
            thresh = 0.8
        else:
            thresh = 1.0

        return [label for label, s in scores.items() if s >= thresh]