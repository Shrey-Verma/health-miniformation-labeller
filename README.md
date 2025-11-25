# Health Misinformation Detection Labeler for Bluesky

## Overview

A hybrid rule-based and embedding-enhanced content moderation system designed to detect and label potentially harmful health-related misinformation in text content. The system uses a sophisticated multi-factor approach combining pattern matching, context analysis, and semantic understanding to identify unsafe medical advice, unverified health claims, and risky health practices.

**Performance Metrics (on 150-post test set)**:
- **F1 Score**: 79.04%
- **Precision**: 85.71%
- **Recall**: 73.33%
- **Exact Match Accuracy**: 77.33%

## Features

- **Five Label Categories**: Detects unverified cures, unsafe medication advice, risky fasting/detox content, unverified supplement claims, and unsafe device usage
- **Context-Aware Detection**: Sentence-level analysis prevents false positives from refutations and legitimate health education
- **Hybrid Approach**: Combines fast rule-based detection with optional embedding-based semantic understanding
- **Multiple Threshold Modes**: Default, conservative, and recall modes for different precision/recall trade-offs
- **Source Verification**: Distinguishes legitimate citations from source misuse
- **Explainable**: Rule-based system provides clear reasoning for labels

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project

Ensure you have the following directory structure:

```
.
├── policy_proposal_labeler.py    # Main entry point
├── health_rules.py                # Core detection logic
├── embedding_context.py           # Optional embedding-based verification
├── evaluate_labeler.py            # Evaluation script
├── data.csv                       # Input data (must have 'text' column)
├── domain_lists/
│   ├── allow_domains.csv          # Trusted health organization domains
│   └── risk_domains.csv           # Known unreliable health websites (optional)
└── requirements.txt               # Dependencies
```

### Step 2: Install Dependencies

The system works with Python standard library only, but optional dependencies enhance performance:

#### Basic Installation (Required)

No additional packages needed! The system works out of the box with Python's standard library.

#### Enhanced Installation (Recommended)

For better sentence tokenization and optional semantic understanding:

```bash
# Install optional dependencies
pip install sentence-transformers nltk

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"
```

**What these provide**:
- **nltk**: Better sentence tokenization for more accurate context detection
- **sentence-transformers**: Semantic embeddings for enhanced context verification (optional layer)

### Step 3: Set Up Domain Lists

The system uses domain lists for context adjustments:

**Create `domain_lists/allow_domains.csv`** (trusted sources):
```
cdc.gov
who.int
nih.gov
nhs.uk
mayoclinic.org
cochranelibrary.com
```

**Create `domain_lists/risk_domains.csv`** (optional, unreliable sources):
```
risk_domain.com
risk_domain.org
risk_domain.info
```

Each line should contain one domain (without `http://` or `https://`). Lines starting with `#` are treated as comments and ignored.

### Step 4: Prepare Input Data

Your input CSV file (`data.csv`) must have at minimum a `text` column. Example:

```csv
post_id,text
1,"Stop taking your insulin, just drink this detox tea instead"
2,"CDC randomized trial shows vaccines are safe"
3,"Dry fast 72h cures diabetes 100%"
```

The system will output predictions in a new column called `predicted_labels`.

## Usage

### Command Line Interface

#### Basic Usage

```bash
python policy_proposal_labeler.py --infile data.csv --outfile preds.csv --mode default
```

#### With Verbose Output (shows score breakdowns)

```bash
python policy_proposal_labeler.py --infile data.csv --outfile preds.csv --mode default --verbose
```

#### Arguments

- `--infile`: Input CSV file with a `text` column (required)
- `--outfile`: Output CSV file with predictions (default: `preds.csv`)
- `--mode`: Threshold mode - `default` (1.0), `conservative` (1.2), or `recall` (0.8)
- `--verbose`: Include score breakdowns in output CSV (shows raw scores for each label category)

**Mode Explanations**:
- **`default`** (threshold = 1.0): Balanced precision and recall
- **`conservative`** (threshold = 1.2): Higher precision, fewer labels (reduces false positives)
- **`recall`** (threshold = 0.8): Higher recall, more labels (catches more edge cases)

### Programmatic Usage

```python
from pathlib import Path
from health_rules import HealthPolicyScorer

# Initialize scorer
scorer = HealthPolicyScorer(domain_dir=Path("domain_lists"))

# Analyze a single text
text = "Stop taking your insulin, just drink this detox tea instead"
labels = scorer.labels_for_text(text, mode="default")
print(labels)  # ['potential-unsafe-medication-advice', 'risky-fasting-detox-content']

# Get detailed scores
scores = scorer.score_text(text)
print(scores)
# {
#   'potential-unverified-cure': 0.0,
#   'potential-unsafe-medication-advice': 1.5,
#   'risky-fasting-detox-content': 1.0,
#   'unverified-supplement-claims': 0.0,
#   'unsafe-device-usage': 0.0
# }
```

### Evaluation

To evaluate predictions against ground truth:

```bash
python evaluate_labeler.py --preds preds.csv --ground_truth data.csv
```

This will output:
- Overall precision, recall, and F1 score
- Per-label breakdown
- Confusion matrix (TP, FP, FN counts)

**Ground truth format**: Your CSV should have a `label_gt` column with pipe-separated labels:
```csv
post_id,text,label_gt
1,"Stop taking insulin","potential-unsafe-medication-advice"
2,"Dry fast 72h cures diabetes","potential-unverified-cure|risky-fasting-detox-content"
3,"CDC shows vaccines are safe",""
```

## How the Labeler Works (Detailed)

### High-Level Architecture

The labeler uses a **three-stage pipeline**:

1. **Pattern Matching**: Identifies potential harmful content using regex patterns
2. **Context-Aware Scoring**: Adjusts scores based on context (refutations, quotes, source citations)
3. **Threshold-Based Labeling**: Applies labels to categories meeting the threshold

### Stage 1: Sentence-Level Pattern Matching

The system first splits input text into sentences for precise analysis:

```python
text = "Stop taking insulin. But first, talk to your doctor."
# Splits into: ["Stop taking insulin.", "But first, talk to your doctor."]
```

Each sentence is then checked against five pattern sets:

#### 1. Unverified Cure Claims (`potential-unverified-cure`)

**Pattern Examples**:
- `"cures cancer"`, `"reverse diabetes"`, `"eliminate autism"`
- `"miracle cure"`, `"cure permanently"`, `"100% cure"`
- `"cure anxiety"`, `"reverse Alzheimer's"`, `"fix arthritis"`

**Matches**: Claims about cures, reversals, or elimination of serious conditions (cancer, diabetes, autism, heart disease, anxiety, arthritis, Alzheimer's, etc.)

**Example**: 
- ✓ `"This herb cures cancer completely"` → Matches
- ✗ `"Treatment helps manage diabetes"` → No match (management vs. cure)

#### 2. Unsafe Medication Advice (`potential-unsafe-medication-advice`)

**Pattern Examples**:
- `"stop taking insulin"`, `"quit antidepressants"`, `"throw away statins"`
- `"avoid all prescription medications"`, `"switch to herbs instead of meds"`
- `"no need for doctor"`, `"cut your dose in half"`
- `"my goal is to stop taking prescriptions"`, `"risk of taking insulin is far greater"`

**Matches**: Content discouraging proper medication use, altering dosages, or avoiding medical consultation

**Example**:
- ✓ `"Stop taking your antidepressants, they don't work"` → Matches
- ✗ `"Talk to your doctor before changing medications"` → No match (safety advice)

#### 3. Risky Fasting/Detox Content (`risky-fasting-detox-content`)

**Pattern Examples**:
- `"72 hour dry fast"`, `"48h water fast"`, `"96 hour fast"`
- `"detox cleanse to drop 10 lbs"`, `"colon cleanse lose weight"`
- `"full body detox cleanse"`, `"dry fast is the only way to cure"`

**Matches**: Extended fasting periods (48+ hours) or dangerous detox/cleanse programs

**Example**:
- ✓ `"72h dry fast cures diabetes"` → Matches
- ✗ `"24 hour broth fast for fun"` → No match (safe duration)

#### 4. Unverified Supplement Claims (`unverified-supplement-claims`)

**Pattern Examples**:
- `"NMN cures all diseases"`, `"NAD+ reverse aging"`, `"peptides fix everything"`
- `"supplements better than prescriptions"`, `"SARMs replace medication"`

**Matches**: Unproven supplements (NMN, NAD+, peptides, SARMs, nootropics) promoted as cures or replacements for medical treatment

**Example**:
- ✓ `"Peptides will cure your diabetes, forget insulin"` → Matches
- ✗ `"I take supplements alongside my medication"` → No match (not replacement)

#### 5. Unsafe Device Usage (`unsafe-device-usage`)

**Pattern Examples**:
- `"use essential oils in nebulizer"`, `"nebulize hydrogen peroxide"`
- `"CGM hack for diabetes"`, `"ignore doctor SpO2 readings"`
- `"colloidal silver in nebulizer"`

**Matches**: Dangerous misuse of medical devices (nebulizers, CGMs, SpO2 monitors)

**Example**:
- ✓ `"Nebulize essential oils to clear your lungs"` → Matches
- ✗ `"Use nebulizer as directed by your doctor"` → No match (proper use)

### Stage 2: Context-Aware Scoring

After pattern matching, each category receives a **base score of 1.0**. The system then applies multiple context adjustments:

#### Step 2.1: Negation Detection

**Purpose**: Prevent false positives from refutations of harmful advice

**How it works**:
- Checks if the sentence negates the harmful pattern
- Looks for negation words: "don't", "never", "false", "wrong", etc.
- Distinguishes actual refutations from harmful advice

**Adjustment**: If negation detected, the pattern is **ignored entirely** (not just score reduction)

**Examples**:
- ✓ `"Don't stop taking your insulin"` → No label (negation detected)
- ✓ `"This claim is false: 'stop taking insulin'"` → No label (refutation)
- ✗ `"Don't trust doctors, use supplements instead"` → Labeled (harmful, not negation)
- ✗ `"CDC is wrong, stop taking insulin"` → Labeled (source misuse, not negation)

**Question Detection**: Questions like "Is it safe to stop taking insulin?" are also treated as negation (not harmful advice).

#### Step 2.2: Quote Detection

**Purpose**: Distinguish quoted misinformation from promoted misinformation

**How it works**:
- Detects balanced quotes (`"..."`)
- Looks for reporting verbs: "say", "claim", "allege", etc.
- Checks for refutation indicators after quotes: "but", "however", "false", "dangerous"

**Adjustment**: `-0.3` per quoted instance (capped at `-0.4` total)

**Examples**:
- ✓ `"Some claim 'stop taking insulin' but that's dangerous"` → Reduced score
- ✓ `"I saw a post saying '72h fast cures diabetes' - this is false"` → Reduced score
- ✗ `"Stop taking insulin because it's bad"` → Full score (not quoted)

#### Step 2.3: Tentative/Hypothetical Language Detection

**Purpose**: Reduce scores for uncertain or conditional language

**How it works**:
- Detects phrases like: "I'll see if", "I'm interested in", "but only with doctor's approval"
- Identifies uncertainty: "confused", "unsure", "uncertain"

**Adjustment**: `-0.5` per instance (capped at `-0.6` total)

**Examples**:
- ✓ `"I'm interested in NMN, but only with my doctor's approval"` → Reduced score
- ✓ `"I'll try a supplement and see if it helps"` → Reduced score
- ✗ `"NMN cures all diseases - buy now!"` → Full score (certainty)

#### Step 2.4: Context Window Analysis

**Purpose**: Check surrounding sentences for context that affects scoring

**How it works**:
- Analyzes 2 sentences before and after the matched sentence
- Looks for refutation patterns, safety advice, or credible sources

**Adjustments**:
- Refutation in context: `-0.5`
- Safety advice ("talk to doctor"): `-0.4`
- Credible sources mentioned: `-0.3`

**Example**:
```
Text: "Some say 'stop taking insulin'. However, 
       you should always talk to your doctor before 
       changing medications."
```
- Pattern matches: "stop taking insulin"
- Context window detects: "talk to your doctor" → `-0.4` adjustment

#### Step 2.5: Domain-Based Adjustments

**Purpose**: Adjust scores based on linked domains

**How it works**:
- Extracts URLs from text
- Checks domains against allow list (CDC, WHO, NIH, etc.) and risk list (unreliable sites)

**Adjustments**:
- **Allow list domain**: `-0.5` (credible source, reduces score)
- **Risk domain**: `+0.3` (unreliable source, increases score)

**Example**:
- `"See cdc.gov article on vaccine safety"` → `-0.5` (legitimate source)
- `"Read this on risk_domain.com about stopping meds"` → `+0.3` (unreliable source)

#### Step 2.6: Source Citation Verification

**Purpose**: Verify if credible sources (CDC, WHO, NIH, etc.) are cited correctly or misused

**How it works**:
- Detects mentions of credible organizations: CDC, WHO, NIH, NHS, EMA, Mayo Clinic, Cochrane
- Checks for legitimate citation patterns: "CDC study", "WHO guidelines", "randomized trial"
- Detects misuse patterns: "CDC is wrong", "despite WHO", "CDC says X but..."

**Adjustments**:
- **Legitimate citation**: `-0.5` (full reduction) or `-0.2` (partial if in refutation context)
- **Source misuse**: `+0.3` (penalty) and sets `source_misuse_detected` flag

**Special Logic**:
- If "CDC says X but WHO says Y", CDC is **not** misused (contradiction is about WHO)
- If source misuse detected, legacy context reductions are **skipped** (don't reduce score for misused sources)

**Examples**:
- ✓ `"CDC randomized trial shows vaccines are safe"` → `-0.5` (legitimate)
- ✗ `"CDC says vaccines cause autism"` → Pattern matched + misuse penalty
- ✗ `"CDC is wrong, stop taking insulin"` → Misuse detected, no context reduction

#### Step 2.7: Embedding-Based Verification (Optional)

**Purpose**: Use semantic similarity to catch nuanced contexts

**How it works** (if `sentence-transformers` installed):
- Encodes text and reference patterns into embeddings
- Calculates cosine similarity to detect refutations
- Verifies source usage semantically

**Adjustments**:
- Refutation detected: `-0.3` to `-0.5` (semantic similarity > 0.7)
- Legitimate source: `-0.3 × confidence`
- Misuse detected: `+0.2 × confidence` (if confidence > 0.5)

**Fallback**: If embeddings unavailable, system uses pattern-based detection only

#### Step 2.8: Stance Amplification

**Purpose**: Increase scores for highly certain or imperative language

**How it works**:
- Counts certainty words: "never", "always", "100%", "guaranteed"
- Counts imperative health patterns: "stop taking", "quit medications"

**Adjustments**:
- Certainty words: `+0.2` per occurrence
- Imperative patterns: `+0.3` per occurrence

**Examples**:
- `"Stop taking insulin 100% guaranteed"` → Base 1.0 + 0.2 (certainty) + 0.3 (imperative) = **1.5**
- `"You might want to consider stopping"` → Base 1.0 (no amplification)

#### Step 2.9: Score Finalization

**Final adjustments**:
1. Apply minimum context adjustment (most conservative)
2. Apply quoted reduction (capped)
3. Apply tentative reduction (capped)
4. Ensure scores don't go negative: `max(0.0, score)`

**Formula**:
```python
final_score = base_score (1.0) 
            + min_context_adjustment  # Most conservative
            - quoted_reduction       # Capped at 0.4
            - tentative_reduction    # Capped at 0.6
            + domain_adjustments
            + citation_adjustments
            + embedding_adjustments
            + stance_amplification
final_score = max(0.0, final_score)  # Non-negative
```

### Stage 3: Threshold-Based Labeling

Scores are compared against the threshold for the selected mode:

- **Default mode** (threshold = 1.0): `score >= 1.0` → Apply label
- **Conservative mode** (threshold = 1.2): `score >= 1.2` → Apply label (higher precision)
- **Recall mode** (threshold = 0.8): `score >= 0.8` → Apply label (higher recall)

**Output**: List of labels as strings, pipe-separated in CSV output

## Example Walkthrough

Let's trace through a complex example:

### Input Text:
```
"I saw a post saying 'stop taking insulin and do a 72h dry fast 
to cure diabetes'. That is dangerous advice - always talk to your 
doctor before changing medications."
```

### Step-by-Step Processing:

1. **Sentence Splitting**:
   - Sentence 1: `"I saw a post saying 'stop taking insulin and do a 72h dry fast to cure diabetes'."`
   - Sentence 2: `"That is dangerous advice - always talk to your doctor before changing medications."`

2. **Pattern Matching** (Sentence 1):
   - ✓ Matches `UNSAFE_MED_PATTERNS`: "stop taking insulin" → `potential-unsafe-medication-advice` base = 1.0
   - ✓ Matches `RISKY_FASTING_DETOX_PATTERNS`: "72h dry fast" → `risky-fasting-detox-content` base = 1.0
   - ✓ Matches `UNVERIFIED_CURE_PATTERNS`: "cure diabetes" → `potential-unverified-cure` base = 1.0

3. **Context Checks** (Sentence 1):
   - ✓ **Quote Detection**: Content is quoted (`'...'`) → `-0.3` per label
   - ✓ **Refutation Detection**: Sentence 2 contains "dangerous advice" → `-0.5` context adjustment
   - ✓ **Safety Advice**: Sentence 2 contains "talk to your doctor" → `-0.4` context adjustment

4. **Score Calculation**:
   ```
   For "potential-unsafe-medication-advice":
   Base: 1.0
   - Quoted: -0.3
   - Context (min of -0.5 and -0.4): -0.5
   Final: 1.0 - 0.3 - 0.5 = 0.2 (below threshold)
   
   For "risky-fasting-detox-content":
   Base: 1.0
   - Quoted: -0.3
   - Context: -0.5
   Final: 0.2 (below threshold)
   
   For "potential-unverified-cure":
   Base: 1.0
   - Quoted: -0.3
   - Context: -0.5
   Final: 0.2 (below threshold)
   ```

5. **Output**: `[]` (no labels) ✓ Correct!

### Another Example (Actually Harmful):

**Input**: `"Stop taking your insulin 100% guaranteed, it's poison!"`

**Processing**:
1. Pattern match: "stop taking insulin" → Base 1.0
2. No negation, quotes, or refutation
3. Stance amplification: "100%" → +0.2, "guaranteed" → +0.2
4. Final score: 1.0 + 0.2 + 0.2 = **1.4**
5. Threshold check: 1.4 >= 1.0 ✓
6. **Output**: `["potential-unsafe-medication-advice"]`

## Performance Details

### Overall Performance (150-post test set)

| Metric | Value |
|--------|-------|
| **F1 Score** | 79.04% |
| **Precision** | 85.71% |
| **Recall** | 73.33% |
| **Exact Match Accuracy** | 77.33% |
| **True Positives** | 66 |
| **False Positives** | 11 (7.3%) |
| **False Negatives** | 24 (16.0%) |

### Per-Label Performance

| Label | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Potential Unverified Cure** | 95.24% | 76.92% | **85.11%** |
| **Unverified Supplement Claims** | 84.21% | 88.89% | **86.49%** |
| **Risky Fasting/Detox Content** | 93.33% | 77.78% | **84.85%** |
| **Potential Unsafe Medication Advice** | 78.57% | 57.89% | **66.67%** |
| **Unsafe Device Usage** | 62.50% | 55.56% | **58.82%** |

**Insights**:
- Strongest: Cure claims and supplements (clear patterns, fewer edge cases)
- Challenges: Medication and device advice (more nuanced contexts)

### Speed Performance

- **Rule-based only**: ~1-5ms per post (very fast)
- **With embeddings**: ~100-200ms per post (includes model loading time)

## Advanced Features

### Context Detection Examples

**Refutation Detection**:
- ✓ `"This claim that vaccines cause autism is misinformation and has been debunked"` → No labels
- ✓ `"Some people say 'stop taking insulin' but this is dangerous advice"` → No labels (quoted + refuted)

**Source Citation**:
- ✓ `"CDC randomized trial shows vaccines are safe"` → No labels (legitimate citation reduces score below threshold)
- ✗ `"CDC says vaccines cause autism"` → Would be flagged (misuse + pattern match)
- ✗ `"CDC is wrong, stop taking insulin"` → Labeled (source misuse doesn't block pattern detection)

**Negation**:
- ✓ `"Don't stop taking your insulin"` → No labels (negation detected)
- ✗ `"Don't trust doctors, use supplements instead"` → Labeled (harmful advice, not negation)
- ✗ `"CDC is wrong, stop taking insulin"` → Labeled (source misuse doesn't negate harmful advice)

**Tentative Language**:
- ✓ `"I'm interested in NMN, but only with my doctor's approval"` → Reduced score (may not meet threshold)
- ✗ `"NMN cures all diseases - buy now!"` → Full score (certainty)

## Design Philosophy

The system uses a **hybrid multi-factor approach**:

- **Layered Detection**: Multiple signal types (patterns, domains, context, language) work together
- **False Positive Prevention**: Context reductions prevent legitimate health education from being flagged
- **Scalable Rules**: Easy to add new patterns or adjust scoring weights
- **Transparent Logic**: Rule-based system makes it clear why content was labeled
- **Semantic Understanding**: Optional embeddings catch nuanced cases

This makes it suitable for policy enforcement where explainability and consistency are important, while still catching sophisticated misinformation through semantic understanding.

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'nltk'`
- **Solution**: This is optional. The system will fall back to simple sentence splitting. To use NLTK: `pip install nltk && python -c "import nltk; nltk.download('punkt')"`

**Issue**: `FileNotFoundError: domain_lists/allow_domains.csv`
- **Solution**: Create the `domain_lists` directory and `allow_domains.csv` file with trusted domains (see installation step 3)

**Issue**: Predictions seem incorrect
- **Solution**: 
  - Check input CSV has a `text` column
  - Try `--verbose` flag to see score breakdowns
  - Verify domain lists are set up correctly
  - Consider using `--mode conservative` for higher precision or `--mode recall` for higher recall

**Issue**: Slow performance
- **Solution**: 
  - Embeddings add ~100-200ms per post. If slow, ensure embeddings are only used when needed
  - Rule-based only should be very fast (~1-5ms per post)

## Contributing

To add new patterns or improve detection:

1. **Add patterns** to the appropriate pattern list in `health_rules.py`:
   - `UNVERIFIED_CURE_PATTERNS`
   - `UNSAFE_MED_PATTERNS`
   - `RISKY_FASTING_DETOX_PATTERNS`
   - `UNVERIFIED_SUPPLEMENT_PATTERNS`
   - `UNSAFE_DEVICE_PATTERNS`

2. **Test on your data** using `evaluate_labeler.py`

3. **Adjust thresholds or scoring weights** as needed in `score_text()` method

4. **Add new context patterns** if needed:
   - `REFUTATION_PATTERNS`: Patterns that indicate misinformation is being debunked
   - `ALLOW_CONTEXT_PATTERNS`: Patterns indicating legitimate health education

## File Structure

```
.
├── policy_proposal_labeler.py    # Main entry point for CSV processing
├── health_rules.py                # Core detection logic (HealthPolicyScorer class)
├── embedding_context.py           # Optional embedding-based verification
├── evaluate_labeler.py            # Evaluation script (precision, recall, F1)
├── data.csv                       # Input data (must have 'text' column)
├── preds.csv                      # Output predictions (generated by labeler)
├── domain_lists/
│   ├── allow_domains.csv          # Trusted health organization domains
│   └── risk_domains.csv           # Known unreliable health websites (optional)
└── requirements.txt               # Dependencies (optional)
```

## License

See assignment requirements for usage terms.
