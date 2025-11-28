# Health Misinformation Detection Labeler for Bluesky

## Group Information
**Group Members:** Shrey Verma(sv528), Akash Basu (ab3334), Fiona Chen

## File Descriptions

- **`policy_proposal_labeler.py`**: Main entry point for the labeler. Provides CLI interface and `run_on_csv()` function that processes input CSV files, applies the moderation logic via `HealthPolicyScorer`, and outputs predictions to a CSV file with labeled results.

- **`health_rules.py`**: Core detection logic containing the `HealthPolicyScorer` class. Implements pattern matching, multi-signal scoring, negation detection, domain verification, and all scoring adjustments. Contains comprehensive regex patterns for detecting harmful health content across four categories.

- **`embedding_context.py`**: Optional semantic enhancement layer. Provides the `EmbeddingContextVerifier` class that uses sentence transformers (`all-MiniLM-L6-v2` model) to detect refutations and verify source usage through semantic similarity. Falls back to rule-based detection if unavailable.

- **`evaluate_labeler.py`**: Evaluation script that compares predictions against ground truth labels. Calculates precision, recall, F1 scores, and provides per-label performance metrics with confusion matrix analysis.

- **`parser.py`**: Script that parses bluesky posts that may fall under medcial misinformation(please note that you will need to provide your own login details to use this script).

- **`analyze.py`**: Script that analyses the evaluation scores and gives an error report of the true positives, false positivs, and false negatives from `data.csv`

- **`data.csv`**: Input dataset containing posts to be labeled. Must have a `text` column at minimum. Can include additional columns like `post_id` and `label_gt` for ground truth evaluation.

- **`domain_lists/allow_domains.csv`**: List of trusted health organization domains (CDC, WHO, NIH, NHS, etc.) used to reduce scores when credible sources are cited.

- **`domain_lists/risk_domains.csv`**: List of known unreliable health websites used to increase scores when questionable sources are cited.

## Overview

A multi-signal rule-based content moderation system designed to detect and label potentially harmful health-related misinformation in text content. The system uses pattern matching with sophisticated scoring adjustments based on context, negation detection, domain verification, and optional semantic embeddings to identify dangerous medical advice, unverified health claims, and risky health practices.

**Performance Metrics (150-post test set)**:
- **F1 Score**: 76.06%
- **Precision**: 95.74%
- **Recall**: 63.08%
- **Exact Match Accuracy**: 67.98%

## Features

- **Four Primary Label Categories**: Detects unverified cure claims, unsafe medication advice, risky fasting/detox content, and unverified supplement claims
- **Multi-Signal Detection**: Combines strong pattern matches with weak signal counting for nuanced detection
- **Advanced Negation Handling**: Distinguishes refutations from promotion of harmful content
- **Domain-Based Verification**: Adjusts scoring based on credible vs. risky domains
- **Optional Embedding Layer**: Semantic similarity detection for enhanced context verification (optional)
- **Explainable Outputs**: Rule-based scoring provides transparency in labeling decisions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Project Structure

Ensure you have the following directory structure:

```
.
├── policy_proposal_labeler.py    # Main entry point
├── health_rules.py                # Core detection logic (HealthPolicyScorer)
├── embedding_context.py           # Optional semantic embedding verification
├── evaluate_labeler.py            # Evaluation script
├── data.csv                       # Input data (must have 'text' column)
├── domain_lists/
│   ├── allow_domains.csv          # Trusted health organization domains
│   └── risk_domains.csv           # Known unreliable health websites
└── requirements.txt               # Dependencies (optional)
```

### Step 2: Install Dependencies

#### Basic Installation (Required)
The system works with Python's standard library only. No additional packages required for basic functionality.

#### Enhanced Installation (Recommended)
For better sentence tokenization and optional semantic understanding:

```bash
pip install sentence-transformers nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

**What these provide**:
- **nltk**: Improved sentence tokenization for better pattern matching
- **sentence-transformers**: Semantic embeddings for refutation detection and source verification (uses `all-MiniLM-L6-v2` model)

### Step 3: Set Up Domain Lists

Create domain lists for context-based scoring adjustments:

**Create `domain_lists/allow_domains.csv`** (trusted health sources):
```
cdc.gov
who.int
nih.gov
nhs.uk
mayoclinic.org
```

**Create `domain_lists/risk_domains.csv`** (unreliable sources - optional):
```
example-risk-domain.com
another-risk-site.org
```

Each line should contain one domain (without `http://` or `https://`). Lines starting with `#` are treated as comments.

### Step 4: Prepare Input Data

Your input CSV must have a `text` column:

```csv
post_id,text
1,"Ivermectin cures COVID-19 100% guaranteed"
2,"CDC study shows vaccines are safe and effective"
3,"72 hour dry fast will cure your diabetes"
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
python policy_proposal_labeler.py --infile data.csv --outfile preds.csv --mode default
```

Or using the Makefile:
```bash
make train
```

#### With Verbose Output
```bash
python policy_proposal_labeler.py --infile data.csv --outfile preds.csv --mode default --verbose
```

The `--verbose` flag adds a `scores` column showing non-zero scores for each category (e.g., `potential-unverified-cure:0.80|potential-unsafe-medication-advice:0.90`).

#### Arguments
- `--infile`: Input CSV file with a `text` column (default: `data.csv`)
- `--outfile`: Output CSV file with predictions (default: `preds.csv`)
- `--mode`: Detection mode - currently supports `balanced` (default)
- `--verbose`: Include score breakdowns in output CSV

### Programmatic Usage

```python
from pathlib import Path
from health_rules import HealthPolicyScorer

# Initialize scorer
scorer = HealthPolicyScorer(domain_dir=Path("domain_lists"), use_embeddings=True)

# Analyze a single text
text = "Ivermectin cures COVID-19"
labels = scorer.labels_for_text(text, mode="balanced")
print(labels)  # ['potential-unsafe-medication-advice']

# Get detailed scores
scores = scorer.score_text(text)
print(scores)
# {
#   'potential-unverified-cure': 0.56,
#   'potential-unsafe-medication-advice': 0.90,
#   'risky-fasting-detox-content': 0.0,
#   'unverified-supplement-claims': 0.0,
# }
```

### Evaluation

To evaluate predictions against ground truth:

```bash
python evaluate_labeler.py --preds preds.csv --ground_truth data.csv
```

Or using the Makefile:
```bash
make eval
```

**Ground truth format**: CSV with a `label_gt` column containing pipe-separated labels:
```csv
post_id,text,label_gt
1,"Ivermectin cures COVID","potential-unsafe-medication-advice"
2,"72h dry fast cures diabetes","potential-unverified-cure|risky-fasting-detox-content"
3,"CDC shows vaccines are safe",""
```

## How the Labeler Works

### Architecture Overview

The labeler uses a **three-stage scoring pipeline**:

1. **Pattern Matching**: Identifies potential harmful content using comprehensive regex patterns
2. **Multi-Signal Scoring**: Assigns base scores and combines multiple weak signals
3. **Context Adjustments**: Applies negation detection, domain verification, and optional embedding-based refinements

### Stage 1: Pattern Matching & Multi-Signal Detection

The system matches text against five pattern categories:

#### 1. Unverified Cure Claims (`potential-unverified-cure`)

Uses a **multi-signal approach** combining strong and weak patterns:

**Strong Patterns** (base score 0.80):
- `"miracle cure"`, `"selling miracle cure"`, `"discovered the miracle cure"`
- `"cured stage 4 cancer"`, `"reversed terminal cancer"`
- `"government hiding cure for cancer"`
- `"natural cure for cancer"`, `"alternative cure for autism"`

**Weak Signals** (accumulated scoring):
- Single weak signal (e.g., `"miracle water"`): 0.35
- Two weak signals: 0.55
- Three or more weak signals: 0.70

**Example**:
- ✓ `"Selling miracle cure for cancer"` → Strong match (0.80)
- ✓ `"Natural miracle treatment for cancer by big pharma"` → 3 weak signals (0.70)
- ✗ `"Treatment helps manage cancer"` → No match

#### 2. Unsafe Medication Advice (`potential-unsafe-medication-advice`)

**Pattern Examples**:
- `"ivermectin cures COVID"`, `"ordered ivermectin"`
- `"bleach cures autism"`, `"chlorine dioxide protocol"`
- `"stop taking insulin"`, `"quit your medication"`
- `"radithor energy tonic"`, `"laetrile cures cancer"`

Base score: **0.90**

**Example**:
- ✓ `"Ordered ivermectin to treat COVID"` → Matches (0.90)
- ✗ `"Talk to your doctor about medication"` → No match

#### 3. Risky Fasting/Detox Content (`risky-fasting-detox-content`)

**Pattern Examples**:
- `"prolonged fasting cures disease"`, `"dry fasting resets immunity"`
- `"fasting 72 hours kills cancer"`, `"water fasting detoxes"`
- `"carnivore diet cures all"`, `"miracle diet cleanses"`

Base score: **0.90**

**Example**:
- ✓ `"72 hour dry fast allegedly cures chronic illness"` → Matches (0.90)
- ✗ `"Intermittent fasting as part of healthy lifestyle"` → No match

#### 4. Unverified Supplement Claims (`unverified-supplement-claims`)

**Pattern Examples**:
- `"kaempferol miracle cure"`, `"apricot kernels cure cancer"`
- `"alkaline water cured my cancer"`, `"miracle spring water"`
- `"turmeric capsules cure cancer"`, `"CBD oil cured autism"`

Base score: **0.90**

**Example**:
- ✓ `"Alkaline water will cure cancer according to oncologist"` → Matches (0.90)
- ✗ `"I take supplements alongside medical treatment"` → No match

### Stage 2: Context Adjustments

After initial scoring, the system applies multiple adjustment layers:

#### Negation & Refutation Detection

**Strong Negation Patterns**:
- `"there is no miracle cure"`, `"no such thing as miracle food"`
- `"FDA alert: no supplement can cure cancer"`
- `"study shows no evidence to support cure claims"`
- `"debunked cure claims"`

**Title Negation** (first sentence only):
- `"Why there's no cure for cancer"`
- `"Alkaline water isn't a cancer cure"`

**Effect**: If strong negation or title negation detected, **all scores reset to 0.0**

**Examples**:
- ✓ `"There is no miracle cure for cancer - this is misinformation"` → No labels (strong negation)
- ✓ `"Why there's no cure for cancer yet"` → No labels (title negation)
- ✗ `"CDC is wrong, stop taking insulin"` → Still labeled (not negation)

#### Category-Specific Score Reduction

When specific categories are detected, the generic `potential-unverified-cure` score is reduced to prevent double-labeling:

- **Medication focus** detected: Reduce cure score by 80% (multiply by 0.2)
- **Supplement focus** detected: Reduce cure score by 70% (multiply by 0.3)
- **Fasting focus** detected: Reduce cure score by 70% (multiply by 0.3)

**Example**:
```
Text: "Ivermectin is a miracle cure for COVID"

Initial scores:
- potential-unverified-cure: 0.80 (miracle cure pattern)
- potential-unsafe-medication-advice: 0.90 (ivermectin pattern)

After reduction:
- potential-unverified-cure: 0.80 × 0.2 = 0.16
- potential-unsafe-medication-advice: 0.90

Final labels: ['potential-unsafe-medication-advice']
```

#### Domain-Based Adjustments

**Allow List Domains** (CDC, WHO, NIH, NHS, etc.):
- Reduces all scores by 0.3 (credible source cited)

**Risk List Domains**:
- Increases all scores by 0.1 (unreliable source cited)

**Examples**:
- `"See cdc.gov for vaccine information"` → -0.3 adjustment
- `"Read on sketchy-health-site.com about stopping meds"` → +0.1 adjustment

#### Source Citation Verification

**Purpose**: Verify if credible sources (CDC, WHO, NIH, etc.) are cited correctly or misused

**How it works**:
- Detects mentions of credible organizations: CDC, WHO, NIH, NHS, EMA, Mayo Clinic, Cochrane
- Checks for legitimate citation patterns: "CDC study", "WHO guidelines", "randomized trial"
- Detects misuse patterns: "CDC is wrong", "despite WHO", "CDC says X but..."


#### Embedding-Based Verification (Optional)

If `sentence-transformers` is installed, semantic similarity detection provides additional refinements:

**Refutation Detection**:
- Compares text to reference patterns like "this is false", "this has been debunked"
- If similarity > 0.75: Reduces all scores by 80% (multiply by 0.2)

**Source Verification**:
- Verifies if credible sources (CDC, WHO, etc.) are cited legitimately
- Detects misuse patterns (e.g., "CDC says X but that's wrong")

**Fallback**: If embeddings unavailable, uses rule-based pattern matching only.

### Stage 3: Threshold-Based Labeling

The system uses **category-specific thresholds**:

| Label | Threshold |
|-------|-----------|
| potential-unsafe-medication-advice | 0.35 |
| potential-unverified-cure | 0.30 |
| risky-fasting-detox-content | 0.35 |
| unverified-supplement-claims | 0.35 |

Labels are applied if: `score >= threshold`

**Final scores are clamped to [0.0, 1.0] range.**

## Example Walkthroughs

### Example 1: Multi-Category Detection with Reduction

**Input**: `"Ivermectin is a miracle cure for COVID-19"`

**Processing**:
1. Pattern matching:
   - Matches `CURE_PATTERNS`: "miracle cure" → `potential-unverified-cure` = 0.80
   - Matches `UNSAFE_MED_PATTERNS`: "ivermectin" + "cure" → `potential-unsafe-medication-advice` = 0.90

2. Category-specific reduction:
   - Medication detected → Reduce cure score by 80%
   - `potential-unverified-cure`: 0.80 × 0.2 = **0.16**
   - `potential-unsafe-medication-advice`: **0.90**

3. Threshold check:
   - 0.16 < 0.30 ✗ (cure label not applied)
   - 0.90 ≥ 0.35 ✓ (medication label applied)

**Output**: `["potential-unsafe-medication-advice"]`

### Example 2: Strong Negation

**Input**: `"There is no miracle cure for cancer. This is a dangerous myth."`

**Processing**:
1. Pattern matching:
   - Would match "miracle cure for cancer"
   
2. Strong negation check:
   - Matches `STRONG_NEGATION_PATTERNS`: "there is no" + "miracle cure"
   - **All scores reset to 0.0**

**Output**: `[]` (no labels)

### Example 3: Domain Adjustment

**Input**: `"Alkaline water cures cancer. Source: cdc.gov/health"`

**Processing**:
1. Pattern matching:
   - Matches `SUPPLEMENT_PATTERNS`: "alkaline water" + "cures cancer" → `unverified-supplement-claims` = 0.90
   
2. Domain extraction:
   - Found domain: `cdc.gov` (in allow list)
   - Adjustment: -0.3
   
3. Final score:
   - 0.90 - 0.3 = **0.60**
   
4. Threshold check:
   - 0.60 ≥ 0.35 ✓

**Output**: `["unverified-supplement-claims"]`

(Note: In reality, CDC wouldn't make this claim, but the system correctly reduces the score due to domain credibility)

### Example 4: Weak Signal Accumulation

**Input**: `"This miracle water will cure all diseases. Big pharma doesn't want you to know."`

**Processing**:
1. Weak signal counting:
   - "miracle" + "cure" = 1 signal
   - "big pharma" + "hiding" = 1 signal
   - Total: 2 weak signals → `potential-unverified-cure` = 0.55

2. Threshold check:
   - 0.55 ≥ 0.30 ✓

**Output**: `["potential-unverified-cure"]`

## Performance Details

### Overall Performance (150-post test set)

| Metric | Value |
|--------|-------|
| **F1 Score** | 76.06% |
| **Precision** | 95.74% |
| **Recall** | 63.08% |
| **Exact Match Accuracy** | 67.98% |
| **True Positives** | 135 |
| **False Positives** | 6 (4.3%) |
| **False Negatives** | 79 (52.7%) |

### Per-Label Performance

| Label | Precision | Recall | F1 Score | TP | FP | FN |
|-------|-----------|--------|----------|----|----|-----|
| **Risky Fasting/Detox Content** | 95.65% | 100.00% | **97.78%** | 22 | 1 | 0 |
| **Unverified Supplement Claims** | 94.74% | 90.00% | **92.31%** | 18 | 1 | 2 |
| **Potential Unsafe Medication Advice** | 100.00% | 62.86% | **77.19%** | 44 | 0 | 26 |
| **Potential Unverified Cure** | 92.73% | 50.00% | **64.97%** | 51 | 4 | 51 |

**Key Insights**:
- **Highest Precision**: Medication advice (100.00%) - zero false positives
- **Highest Recall**: Fasting/detox content (100.00%) - catches all instances
- **Strongest Overall**: Fasting/detox (97.78% F1) - clear patterns, fewer edge cases
- **Challenge Area**: Unverified cure claims (50% recall) - nuanced language requires improvement

### Error Analysis

**False Negatives (Main Challenge)**:
- Cure claims: 51 missed instances (subtle phrasing, implicit claims)
- Medication advice: 26 missed instances (indirect recommendations)

**False Positives (Minimal)**:
- Total: 6 false positives across all categories
- Strong precision reflects effective negation and context detection

## Advanced Features

### Skip Patterns
The system automatically skips posts that are clearly unrelated to health misinformation:
- Posts about personal news without health content
- Obvious bot malfunctions or spam
- Posts with specific non-health keywords (bootlicking, POW, etc.)

### Multi-Signal Detection Philosophy
Rather than relying on single strong patterns, the system:
- Combines multiple weak signals to detect nuanced misinformation
- Uses category-specific score reduction to prevent over-labeling
- Applies context-aware adjustments to distinguish education from promotion

### Transparency
The `--verbose` flag outputs raw scores for each category, making it easy to:
- Debug why a label was or wasn't applied
- Understand which patterns triggered detection
- Fine-tune thresholds based on use case

## Design Philosophy

The system prioritizes:

- **High Precision Over Recall**: 95.74% precision ensures minimal false positives, suitable for content moderation
- **Explainable Decisions**: Rule-based scoring shows exactly why content was flagged
- **Context Awareness**: Strong negation detection prevents flagging health education
- **Extensibility**: Easy to add new patterns or adjust scoring weights
- **Graceful Degradation**: Works without optional dependencies (NLTK, embeddings)

This makes it suitable for policy enforcement where false positives are costly and explainability is important.

## Troubleshooting

### Common Issues

**Issue**: `Warning: sentence-transformers not available`
- **Solution**: This is optional. The system falls back to rule-based detection. To use embeddings: `pip install sentence-transformers`

**Issue**: `FileNotFoundError: domain_lists/allow_domains.csv`
- **Solution**: Create the `domain_lists` directory and `allow_domains.csv` file (see Step 3)

**Issue**: All predictions are empty
- **Solution**: 
  - Check that input CSV has a `text` column
  - Verify text contains health-related content
  - Try `--verbose` to see if scores are being generated but not meeting thresholds

**Issue**: Too many false positives
- **Solution**: 
  - Verify domain lists are set up correctly
  - Check that strong negation patterns are working
  - Consider adjusting thresholds in `health_rules.py`

## Contributing

To improve detection:

1. **Add patterns** to `health_rules.py`:
   - `CURE_PATTERNS`: Strong cure claim patterns
   - `WEAK_CURE_SIGNALS`: Accumulative signals for cure claims
   - `UNSAFE_MED_PATTERNS`: Dangerous medication advice
   - `SUPPLEMENT_PATTERNS`: Unverified supplement claims
   - `FASTING_PATTERNS`: Risky fasting/detox content

2. **Add negation patterns**:
   - `STRONG_NEGATION_PATTERNS`: Global refutation indicators
   - `TITLE_NEGATION_PATTERNS`: First-sentence negations

3. **Adjust thresholds** in `labels_for_text()` method based on precision/recall needs

4. **Test changes** using `evaluate_labeler.py` with ground truth data

## File Structure

```
.
├── policy_proposal_labeler.py    # CLI entry point, CSV processing
├── health_rules.py                # HealthPolicyScorer class, core logic
├── embedding_context.py           # EmbeddingContextVerifier (optional)
├── evaluate_labeler.py            # Evaluation metrics script
├── data.csv                       # Input data
├── preds.csv                      # Output predictions
├── domain_lists/
│   ├── allow_domains.csv          # Trusted domains (CDC, WHO, etc.)
│   └── risk_domains.csv           # Unreliable domains
└── Makefile                       # Shortcuts (make train, make eval)
```

## License

See assignment requirements for usage terms.
