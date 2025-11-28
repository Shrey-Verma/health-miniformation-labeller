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
    def sent_tokenize(text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

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

UNSAFE_MEDICATIONS = [
    "ivermectin", "bleach", "chlorine dioxide", "mms", 
    "lupron", "colloidal silver", "methylene blue", "radium",
    "laetrile", "blue fabric dye", "fabric dye", "hydroxychloroquine",
    "black salve", "coffee enema", "mercury", "bear bile", "radithor"
]

SUPPLEMENTS = [
    "kaempferol", "apricot kernel", "amygdalin", "chitosan", "turmeric", 
    "baking soda", "organic smoothie", "alkaline water", "miracle water",
    "cbd oil", "essential oil", "virgin coconut oil", "black seed oil",
    "rose syrup", "rooh afza", "bamboo salt", "hydrogen water"
]

UNSAFE_MED_PATTERNS = [
    r"(?i)\bivermectin\b.{0,100}\b(cure|treat|miracle|works for|taking for|ordered|recommend|cancer|covid|shown a post|feels cured|high doses)\b",
    r"(?i)\border(ed)?\s+ivermectin\b",
    r"(?i)\b(shown a post|opened it)\b.{0,50}\bivermectin\b",
    r"(?i)\b(bleach|chlorine dioxide|MMS)\b.{0,100}\b(cure|cures|drink|miracle|protocol|peddling|sells|huckster|quack|trump|rfk|autism|cancer|covid|deadly|solution)\b",
    r"(?i)\bit'?s\s+(bleach|chlorine|deadly)\b",
    r"(?i)\bthis\s+is\s+(?:the\s+)?.miracle\s+cure.\b.{0,50}\b(it'?s\s+)?bleach\b",
    r"(?i)\bwill\s+(announce|be\s+peddling)\b.{0,50}\bmiracle\s+cure\b.{0,50}\bit'?s\s+bleach\b",
    r"(?i)\b(blue\s+fabric\s+dye)\b.{0,80}\bmiracle\s+cure\b.{0,50}\b(cancer|stage\s+4)\b",
    r"(?i)\bmeet\s+\w+\b.{0,50}\b(quack|huckster|sells)\b.{0,50}\b(?:the\s+)?.miracle\s+cure.\b.{0,50}\b(chlorine|bleach)\b",
    r"(?i)\bsells?\s+(?:the\s+)?.miracle\s+cure.\b.{0,50}\b(chlorine|bleach)\b",
    r"(?i)\bpeople\s+used\s+to\s+drink\b.{0,30}\bradium\b",
    r"(?i)\bradium\b.{0,80}\b(marketed|tonic|health|cure|drink)\b",
    r"(?i)\bvaccines\s+cause\b.{0,100}\bradium\b.{0,50}\bmiracle\s+cure\b",
    r"(?i)\blaetrile\b.{0,80}\b(cancer|cure|miracle|treatment|recall|rubes|fleeced|eschewed)\b",
    r"(?i)\blupron\b.{0,50}\b(autism|cure|miracle)\b",
    r"(?i)\bcolloidal\s+silver\b.{0,50}\b(cure|antibiotic|treat)\b",
    r"(?i)\bmethylene\s+blue\b.{0,50}\b(cancer|cure|aging)\b",
    r"(?i)\b(stop|quit|throw\s+away|ditch|don't\s+need)\s+(taking\s+)?(your\s+)?\b(insulin|medication|meds|prescription)\b",
    r"(?i)\b(this\s+drug|the\s+drug|one\s+drug|form\s+of\s+birth\s+control)\b.{0,100}\b(touted|being\s+sold|marketed|being\s+sold\s+as)\b.{0,50}\b(cure|miracle)\b",
    r"(?i)\b(doctor|someone)\s+on\s+tiktok\b.{0,50}\bnew\s+supplement\b.{0,50}\bcure\b.{0,50}\b(COVID|cancer)\b",
    r"(?i)\b(took|taking)\s+high\s+doses\b.{0,50}\bivermectin\b",
    r"(?i)\b(friend|guy|person)\s+(said|took)\b.{0,50}\b(high\s+doses|ivermectin)\b.{0,50}\b(feels\s+cured|cured)\b",
    r"(?i)\b(instagram\s+)?(?:influencer|social\s+media\s+post|tiktok|youtube)\b.{0,80}\b(promotes?|claims?|advertises?)\b.{0,80}\bmiracle\s+(serum|pills?|device|mushroom\s+extract|anti-aging\s+pills|injection|capsules?|CBD\s+(?:capsules?|oil)|vitamin)\b",
    r"(?i)\bmiracle\s+(device|mushroom|mushroom\s+extract|injection|capsules?|CBD\s+(?:capsules?|oil)|vitamin|serum)\b.{0,80}\b(cure|cures|for|stops)\b.{0,50}\b(diseases?|stage\s+4|cancer|COVID|autism)\b",
    r"(?i)\b(someone|person)\s+(?:online\s+)?claimed\b.{0,80}\b(herbal\s+concoction|new\s+\w+)\b.{0,80}\bcures?\b.{0,50}\b(COVID|cancer)\b",
    r"(?i)\bsome\s+(?:americans|people)\s+believe\b.{0,80}\b(blue\s+fabric\s+dye|fabric\s+dye)\b.{0,50}\bmiracle\s+cure\b",
    r"(?i)\b(hydroxychloroquine|hcq)\b.{0,80}\b(cure|cures|miracle)\b.{0,50}\b(covid|cancer)\b",
    r"(?i)\bskip\s+(?:your\s+)?(vaccine|chemo|treatment)\b.{0,50}\b(use|try)\b.{0,50}\b(this|natural)\b",
    # specific unsafe patterns
    r"(?i)\b(bear\s+bile)\b.{0,80}\b(cure|selling|prescribed)\b.{0,50}\b(liver\s+disease|arthritis)\b",
    r"(?i)\b(coconut\s+oil\s+enema)\b.{0,80}\b(cure|claiming)\b.{0,50}\b(parkinson)\b",
    r"(?i)\b(radithor|radioactive\s+water)\b.{0,80}\b(energy\s+tonic|cure|sold\s+as)\b",
    r"(?i)\btrepanning\b.{0,80}\b(cure|drilling\s+holes)\b.{0,50}\b(headache|epilepsy|mental\s+illness)\b",
]

SUPPLEMENT_PATTERNS = [
    r"(?i)\bkaempferol\b.{0,80}\b(is\s+great|miracle|named after|naturalist|10/10|used for cancer treatment)\b",
    r"(?i)\b(alkaline\s+water|miracle\s+(?:spring\s+)?water)\b.{0,80}\b(cured|erase\s+debt|implied)\b",
    r"(?i)\bmiracle\s+(?:spring\s+)?water\b.{0,80}\b(airtime|evangelist|selling|advertising)\b",
    r"(?i)\b(?:ofcom|religious\s+(?:TV\s+)?channel)\s+(?:has\s+)?fined\b.{0,100}\bmiracle\s+(?:spring\s+)?water\b.{0,80}\bcure\b",
    r"(?i)\b(?:fined|fine).*\bmiracle\s+(?:spring\s+)?water\b.{0,80}\b(?:cure|claims?)\b",
    r"(?i)^Can\s+[Aa]lkaline\s+water\s+(?:prevent\s+or\s+)?cure\s+cancer",
    r"(?i)\b[Aa]lkaline\s+water\s+(?:prevent\s+or\s+)?cure\s+cancer.*\b(?:oncologist|Times\s+of\s+India|explains)\b",
    r"(?i)\b(apricot\s+kernels?|amygdalin)\b.{0,80}\b(cure|miracle|kept\s+bags|claiming)\b",
    r"(?i)\b(smoothie|juice|organic\s+smoothie)\b.{0,100}\b(cure|could\s+cure|thought.*cure|drinking)\b.{0,50}\b(cancer|pancreatic)\b",
    r"(?i)\bsteve\s+jobs\b.{0,100}\b(cure|thought.*cure)\b.{0,50}\b(smoothie|organic|drinking)\b",
    r"(?i)\b(turmeric|baking\s+soda)\b.{0,80}\bcure\b.{0,50}\bcancer\b",
    r"(?i)\bturmeric\s+capsules\b.{0,80}\b(?:cure|prevent)\b.{0,50}\bcancer\b",
    r"(?i)\bchitosan\b.{0,50}\bcancer\b",
    r"(?i)\bgiving\s+up\s+sugar\b.{0,50}\b(cure|would\s+cure)\b",
    r"(?i)\bsupplement\b.{0,80}\b(cure|will\s+cure)\b.{0,50}\b(aging|arthritis|inflammation)\b",
    r"(?i)\bCBD\s+oil\b.{0,50}\b(cure|cured)\b.{0,50}\b(cancer|autism|epilepsy)\b",
    r"(?i)\bessential\s+oils?\b.{0,50}\b(cure|cured)\b.{0,50}\b(cancer|disease)\b",
    # geographic/cultural supplements 
    r"(?i)\b(virgin\s+coconut\s+oil)\b.{0,80}\b(?:marketed\s+as|booming\s+business|preventing|curing)\b.{0,50}\b(lupus)\b",
    r"(?i)\b(black\s+seed\s+oil)\b.{0,80}\b(?:promoting|cure|promoted\s+as)\b.{0,50}\b(asthma|diabetes|hypertension)\b",
    r"(?i)\b(?:rooh\s+afza|rose\s+syrup)\b.{0,80}\b(?:claiming|mixed\s+with|cure|cures)\b.{0,50}\b(gastritis|ulcer)\b",
    r"(?i)\b(bamboo\s+salt)\b.{0,80}\b(?:prevent|claiming|prevents)\b.{0,50}\b(cancer|diabetes)\b",
    r"(?i)\b(hydrogen\s+water)\b.{0,80}\b(?:prevent|cure|reverse)\b.{0,50}\b(alzheimer|aging)\b",
]

FASTING_PATTERNS = [
    r"(?i)\b(prolonged\s+(?:water\s+)?fasting|dry\s+fast(?:ing)?|water\s+fast(?:ing)?|extended\s+fasting)\b.{0,100}\b(cure|kills?.*cancer|reset|heal|detox|claim|chronic\s+illness|multiple\s+illnesses|dangerous|allegedly|supposedly|cures\s+disease)\b",
    r"(?i)\bfasting\s+(?:for\s+)?\d+\s+(hours?|days?)\b.{0,100}\b(cure|kills?.*cancer|reset|heal|detox|claim|prevent|supposedly|straight|allegedly|detoxes)\b",
    r"(?i)\b(tried|anyone\s+tried)\s+fasting\b.{0,70}\d+\s+hours?\b.{0,50}\b(reset|kill.*cancer|supposedly|straight)\b",
    r"(?i)\bintermittent\s+fasting\b.{0,100}\b(reset|cure|allegedly|supposedly|resets\s+immunity|cures\s+disease)\b",
    r"(?i)\b(carnivore\s+diet|hunza\s+diet)\b.{0,100}\b(cure\s+all|cancer\s+free|long\s+life|miracle|supposedly|village|cured)\b",
    r"(?i)\b(his\s+diets?|extreme\s+diet|detox\s+diets?|juicing\s+diet)\b.{0,80}\b(cure\s+all\s+disease|could\s+cure\s+all|claim\s+to\s+cure|cure\s+everything|detox|prevent|reset|allegedly|supposedly)\b",
    r"(?i)\bmiracle\s+diet\b.{0,50}\b(cleanse|kill.*cancer|cure)\b",
]

CURE_PATTERNS = [
    #  always flag these
    r"(?i)\bmiracle\s+(?:cancer\s+)?cure\b",
    r"(?i)\b(selling|peddling|hawking)\s+(?:a\s+)?miracle\s+cure\b",
    r"(?i)\b(discovered|found)\s+(?:the\s+)?miracle\s+cure\b",
    r"(?i)\bscam\b.{0,50}\bmiracle\s+cure\b",
    r"(?i)\b(trump|rfk|kennedy)\b.{0,80}\b(miracle\s+cure|cure.*autism|peddling|hawking)\b",
    r"(?i)\b(hoxsey|conman|charlatan)\b.{0,50}\bmiracle\s+cure\b",
    r"(?i)\bgovernment\b.{0,50}\b(hiding|suppressing)\b.{0,50}\bcure\b.{0,50}\bcancer\b",
    r"(?i)\b(lured|preying\s+on)\b.{0,50}\b(desperate|vulnerable)\b.{0,50}\bmiracle\s+cure\b",
    
    # require disease context
    r"(?i)\b(cure|cured|cures)\s+(?:my\s+|your\s+|his\s+|her\s+)?(stage\s+4|terminal|pancreatic)\s+(cancer|leukemia)\b",
    r"(?i)\b(friend|neighbor|coworker|guy|someone)\b.{0,80}\b(claimed|swears|said|told)\b.{0,80}\b(cured|reversed|healed)\b.{0,50}\b(stage\s+4|terminal|pancreatic|cancer)\b",
    r"(?i)\b(herbal|mushroom|rare\s+fungus|natural)\b.{0,80}\b(cure|cured|reversed)\b.{0,50}\b(stage\s+4|terminal|pancreatic|cancer)\b",
    r"(?i)\bprefer(?:red)?\s+(?:the\s+)?natural\s+(?:miracle\s+)?cure\b.{0,50}\b(chemotherapy|treatment|medicine)\b",
    
    #  disease + cure claims
    r"(?i)\b(cure|cured)\b.{0,50}\b(cancer|diabetes|autism|alzheimer)\b.{0,50}\b(naturally|natural|miracle|alternative)\b",
    r"(?i)\b(natural|alternative)\b.{0,50}\b(cure|treatment)\b.{0,50}\b(cancer|diabetes|autism)\b",
    r"(?i)\b(someone|person|doctor)\b.{0,80}\b(claimed|claims|says)\b.{0,80}\b(cure|cures)\b.{0,50}\b(cancer|covid|autism)\b",
    
    # question patterns with specific disease
    r"(?i)\b(can|could|does|will)\b.{0,30}\b(cure|prevent)\b.{0,30}\b(cancer|stage\s+4|terminal)\b",
    r"(?i)\b(heard|read|saw)\b.{0,50}\b(cure|cures)\b.{0,50}\b(cancer|diabetes|autism)\b",
    
    # historical patterns
    r"(?i)\b(1950s|1960s|1970s)\b.{0,80}\bmiracle\s+cure\b",
    r"(?i)\bsteaming\s+compost\b.{0,50}\bmiracle\s+cure\b",
    
    # additional patterns
    r"(?i)\b(?:this|that|the)\s+(?:will|can|could)\s+cure\s+(?:cancer|diabetes|autism)\b",
    r"(?i)\bfinally\s+(?:found|discovered)\s+(?:a\s+)?cure\b.{0,50}\b(cancer|diabetes)\b",
    r"(?i)\b(?:eliminates?|destroys?|kills?)\s+cancer\s+cells?\b.{0,50}\bnaturally\b",
    r"(?i)\breverse\s+(?:diabetes|cancer|alzheimer)\s+with\b",
    r"(?i)\b(?:amazing|incredible|revolutionary)\s+(?:new\s+)?(?:cure|treatment|discovery)\b.{0,50}\b(cancer|diabetes)\b",
]

WEAK_CURE_SIGNALS = [
    r"(?i)\bmiracle\b.{0,30}\b(cure|treatment|water|serum)\b",
    r"(?i)\bcure\s+(?:for\s+)?(?:stage\s+4|terminal|pancreatic)\b",
    r"(?i)\b(?:natural|alternative)\s+(?:cure|treatment)\b.{0,50}\b(?:cancer|diabetes|autism)\b",
    r"(?i)\b(?:big\s+pharma|government|they)\s+(?:don'?t\s+want|hiding|suppressing)\b.{0,50}\bcure\b",
    r"(?i)\b(?:selling|hawking|peddling)\b.{0,50}\b(?:cure|miracle|treatment)\b",
    r"(?i)\b(?:claims?|swears|promised)\s+(?:to\s+)?cure\b",
]

STRONG_NEGATION_PATTERNS = [
    # Strong negation about cures
    r"(?i)\b(there\s+is\s+no|no\s+such\s+thing\s+as)\b.{0,50}\bmiracle\b.{0,50}\b(cure|food|supplement|that\s+will\s+cure)\b",
    r"(?i)\bno\s+miracle\s+(?:food|supplement|cure)\b.{0,50}\bwill\s+cure\b",
    r"(?i)\b(?:it'?s\s+just\s+so\s+)?dangerous\s+and\s+wrong\b.{0,50}\bno\s+miracle\b",
    r"(?i)\bwhy\s+there'?s\s+no\s+cure\s+for\s+cancer\b",
    
    # scientific/regulatory negation - NEW!
    r"(?i)\b(?:FDA|NIH|Lancet|Cochrane|peer-reviewed)\b.{0,80}\b(?:no\s+evidence|no\s+correlation|no\s+data|does\s+not\s+cure|are\s+not\s+approved)\b",
    r"(?i)\b(?:FDA|NIH)\s+(?:alert|advisory|guidance|statement)\b.{0,50}\b(?:no\s+supplement|cannot\s+legally\s+claim)\b",
    r"(?i)\bno\s+clinical\s+evidence\b.{0,50}\b(?:cure|curing|preventing)\b",
    r"(?i)\b(?:study|research|journal|article)\b.{0,50}\b(?:no\s+evidence|does\s+not\s+cure|anti-inflammatory\s+properties\s+but\s+does\s+not)\b",
    
    # other negation patterns
    r"(?i)\bno\s+evidence\b.{0,30}\b(?:support|show|prove)\b.{0,30}\bcure\b",
    r"(?i)\bdebunk(?:ed|ing)\b.{0,50}\b(?:cure|miracle|claim)\b",
    r"(?i)\bdon'?t\s+(?:believe|fall\s+for)\b.{0,50}\b(?:cure|miracle|claim)\b",
]

TITLE_NEGATION_PATTERNS = [
    r"(?i)^.{0,80}\bisn'?t\s+a\s+(?:cancer\s+)?cure\b",
    r"(?i)^.{0,80}\bis\s+not\s+a\s+cure\b",
    r"(?i)^.{0,80}\bwhy\s+there'?s\s+no\s+cure\b",
    r"(?i)^.{0,80}\bno\s+cure\s+for\b",
]

SKIP_PATTERNS = [
    r"^(?=.*\b(bootlicking|POW|loser.*funeral)\b)(?!.*\b(cure|miracle)\b).{0,300}$",
    r"^(?=.*\b(personal\s+news|disappear)\b)(?!.*\b(cure|miracle)\b).{0,150}$",
    r"^(?=.*\bmalfunctioning.*congratulations.*cancer\b)",
    r"(?i)^.{0,100}\b(personal\s+news|just\s+in\s+case|disappear\s+into\s+the\s+ether)\b",
    r"(?i)\bbot.*malfunctioning\b",
    r"(?i)\bcongratulations.*cancer.*haha\b",
]
#load domain list from embedding context
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

#extract from embedding context 
def extract_domains(text: str):
    URL_REGEX = re.compile(r"https?://\S+")
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
    return any(re.search(p, text, flags=re.IGNORECASE | re.DOTALL) for p in patterns)

def count_matches(patterns, text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE | re.DOTALL))

def contains_substance(text: str, substances: list) -> bool:
    text_lower = text.lower()
    return any(substance in text_lower for substance in substances)

def should_skip_entirely(text: str) -> bool:
    return any_match(SKIP_PATTERNS, text)

def has_strong_negation(text: str) -> bool:
    return any_match(STRONG_NEGATION_PATTERNS, text)

def has_title_negation(text: str) -> bool:
    sentences = sent_tokenize(text)
    if not sentences:
        return False
    first_sentence = sentences[0].strip()
    if len(first_sentence) > 150:
        return False
    return any_match(TITLE_NEGATION_PATTERNS, first_sentence)

#get score based off matches and weights specificed 
class HealthPolicyScorer:
    def __init__(self, domain_dir: Path | None = None, use_embeddings: bool = True):
        domain_dir = domain_dir or Path("domain_lists")
        self.allow_domains = load_domain_list(domain_dir / "allow_domains.csv")
        self.risk_domains = load_domain_list(domain_dir / "risk_domains.csv")
        
        self.embedding_verifier = None
        if use_embeddings and EMBEDDING_VERIFIER_AVAILABLE and EmbeddingContextVerifier:
            try:
                self.embedding_verifier = EmbeddingContextVerifier(use_embeddings=True)
            except Exception as e:
                print(f"Warning: Could not initialize embedding verifier: {e}")

    def score_text(self, text: str) -> Dict[str, float]:
        t = text.strip()
        scores: Dict[str, float] = {
            LABEL_UNVERIFIED_CURE: 0.0,
            LABEL_UNSAFE_MED_ADVICE: 0.0,
            LABEL_RISKY_FASTING_DETOX: 0.0,
            LABEL_UNVERIFIED_SUPPLEMENT: 0.0,
            LABEL_UNSAFE_DEVICE_USE: 0.0,
        }

        if should_skip_entirely(t):
            return scores

        if has_strong_negation(t):
            return scores
            
        if has_title_negation(t):
            return scores

        # MULTI-SIGNAL DETECTION for cure patterns
        cure_strong_match = any_match(CURE_PATTERNS, t)
        weak_signal_count = count_matches(WEAK_CURE_SIGNALS, t)
        
        if cure_strong_match:
            scores[LABEL_UNVERIFIED_CURE] = 0.80
        elif weak_signal_count >= 3:
            scores[LABEL_UNVERIFIED_CURE] = 0.70
        elif weak_signal_count == 2:
            scores[LABEL_UNVERIFIED_CURE] = 0.55
        elif weak_signal_count == 1:
            scores[LABEL_UNVERIFIED_CURE] = 0.35

        # check specific categories
        unsafe_med_match = any_match(UNSAFE_MED_PATTERNS, t)
        supplement_match = any_match(SUPPLEMENT_PATTERNS, t)
        fasting_match = any_match(FASTING_PATTERNS, t)
        
        if unsafe_med_match:
            scores[LABEL_UNSAFE_MED_ADVICE] = 0.9
            # Strongly reduce unverified-cure for medication focus
            if scores[LABEL_UNVERIFIED_CURE] > 0:
                scores[LABEL_UNVERIFIED_CURE] *= 0.2

        if supplement_match:
            scores[LABEL_UNVERIFIED_SUPPLEMENT] = 0.9
            # Reduce unverified-cure for supplement focus
            if scores[LABEL_UNVERIFIED_CURE] > 0:
                scores[LABEL_UNVERIFIED_CURE] *= 0.3

        if fasting_match:
            scores[LABEL_RISKY_FASTING_DETOX] = 0.9
            # Reduce unverified-cure for fasting focus
            if scores[LABEL_UNVERIFIED_CURE] > 0:
                scores[LABEL_UNVERIFIED_CURE] *= 0.3

        # embedding adjustment
        if self.embedding_verifier:
            try:
                is_refutation, confidence = self.embedding_verifier.detect_refutation_context(t)
                if is_refutation and confidence > 0.75:
                    for k in scores:
                        scores[k] *= 0.2
            except Exception:
                pass

        # domain adjustments
        domains = extract_domains(t)
        if domains:
            if any(d in self.risk_domains for d in domains):
                for k in scores:
                    if scores[k] > 0:
                        scores[k] = min(1.0, scores[k] + 0.1)
            if any(d in self.allow_domains for d in domains):
                for k in scores:
                    scores[k] = max(0, scores[k] - 0.3)

        # clamp
        for k in scores:
            scores[k] = min(1.0, max(0.0, scores[k]))

        return scores
#balance weights 
    def labels_for_text(self, text: str, mode: str = "balanced") -> List[str]:
        scores = self.score_text(text)

        thresholds = {
            LABEL_UNSAFE_MED_ADVICE: 0.35,
            LABEL_UNVERIFIED_CURE: 0.30,
            LABEL_RISKY_FASTING_DETOX: 0.35,
            LABEL_UNVERIFIED_SUPPLEMENT: 0.35,
            LABEL_UNSAFE_DEVICE_USE: 0.35,
        }

        return [label for label, s in scores.items() if s >= thresholds.get(label, 0.30)]