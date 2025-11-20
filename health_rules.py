from __future__ import annotations
from typing import List, Dict, Set
import re
from pathlib import Path
from urllib.parse import urlparse

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
    r"\b(double|triple|cut|halve)\s+your\s+(dose|dosage)\b",
    r"\b(no need|don['’]t need)\b.*\b(prescription|doctor)\b",
]

RISKY_FASTING_DETOX_PATTERNS = [
    r"\b(dry fast|water fast)\b.*\b(48|72|96|120)\s*(h|hours?)\b",
    r"\bdetox (tea|cleanse|flush)\b.*\b(weight|fat|toxins?)\b",
    r"\bcolon cleanse\b.*\b(lose|drop)\s+\d+\s*(kg|kgs|pounds|lbs)\b",
]

UNVERIFIED_SUPPLEMENT_PATTERNS = [
    r"\bNMN|NAD\+|peptides?|SARMs?|nootropics?\b.*\b(cures?|fix(es|ed)|revers(es|e))\b",
    r"\bjust take\b.*\b(supplements?|herbs?|pills?)\b.*\b(no need|instead)\b.*\b(doctor|medication|treatment)\b",
]

UNSAFE_DEVICE_PATTERNS = [
    r"\bnebuliz(e|er)\b.*\b(essential oils?|hydrogen peroxide|vinegar)\b",
    r"\bCGM\b.*\b(hack|game|cheat)\b.*\b(diabetes|insulin)\b",
    r"\bSpO2\b.*\b(no need\b.*\bdoctor|ignore\b.*\bdoctor)\b",
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


class HealthPolicyScorer:
    def __init__(self, domain_dir: Path | None = None):
        domain_dir = domain_dir or Path("domain_lists")
        self.allow_domains = load_domain_list(domain_dir / "allow_domains.csv")
        self.risk_domains = load_domain_list(domain_dir / "risk_domains.csv")

    def score_text(self, text: str) -> Dict[str, float]:
        t = text.strip()
        scores: Dict[str, float] = {
            LABEL_UNVERIFIED_CURE: 0.0,
            LABEL_UNSAFE_MED_ADVICE: 0.0,
            LABEL_RISKY_FASTING_DETOX: 0.0,
            LABEL_UNVERIFIED_SUPPLEMENT: 0.0,
            LABEL_UNSAFE_DEVICE_USE: 0.0,
        }

        if any_match(UNVERIFIED_CURE_PATTERNS, t):
            scores[LABEL_UNVERIFIED_CURE] += 1.0
        if any_match(UNSAFE_MED_PATTERNS, t):
            scores[LABEL_UNSAFE_MED_ADVICE] += 1.0
        if any_match(RISKY_FASTING_DETOX_PATTERNS, t):
            scores[LABEL_RISKY_FASTING_DETOX] += 1.0
        if any_match(UNVERIFIED_SUPPLEMENT_PATTERNS, t):
            scores[LABEL_UNVERIFIED_SUPPLEMENT] += 1.0
        if any_match(UNSAFE_DEVICE_PATTERNS, t):
            scores[LABEL_UNSAFE_DEVICE_USE] += 1.0

        domains = extract_domains(t)
        if domains:
            if any(d in self.risk_domains for d in domains):
                for k in scores:
                    scores[k] += 0.3
            if any(d in self.allow_domains for d in domains):
                for k in scores:
                    scores[k] -= 0.5

        if any_match(ALLOW_CONTEXT_PATTERNS, t) or any_match(REFUTATION_PATTERNS, t):
            for k in scores:
                scores[k] -= 0.7

        certainty_hits = count_matches(CERTAINTY_WORDS, t)
        imperative_hits = count_matches(IMPERATIVE_HEALTH_PATTERNS, t)
        stance_boost = 0.2 * certainty_hits + 0.3 * imperative_hits
        if stance_boost:
            for k in scores:
                if scores[k] > 0:
                    scores[k] += stance_boost

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
