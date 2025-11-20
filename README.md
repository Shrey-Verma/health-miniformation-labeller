# Labeler System Explanation

## Overview

This is a rule-based content moderation system designed to detect and label potentially harmful health-related misinformation in text content. The system analyzes text and assigns one or more labels when it detects patterns that suggest unsafe medical advice, unverified health claims, or risky health practices.

## System Architecture

The labeler consists of two main components:

1. **CSV Processing System**: Reads text from CSV files, analyzes each entry, and outputs predictions to a new CSV file
2. **Bluesky Integration System**: Designed to work with Bluesky social media posts (currently in development)

## How It Works

### Step 1: Pattern Matching

The system first scans the input text for specific patterns that indicate potentially problematic health content. It looks for five different categories of concerns:

#### 1. Unverified Cure Claims
The system detects claims about cures, reversals, or elimination of serious medical conditions (cancer, diabetes, autism, heart disease). It flags language suggesting miracle treatments or claims of treatments with no side effects for any disease.

#### 2. Unsafe Medication Advice
The system identifies content that discourages proper medication use, such as:
- Instructions to stop taking prescribed medications (insulin, statins, antidepressants, blood pressure medications)
- Advice to alter medication dosages without medical supervision
- Claims that prescriptions or doctor consultations are unnecessary

#### 3. Risky Fasting and Detox Content
The system flags potentially dangerous fasting or detox advice, including:
- Extended fasting periods (dry fasts or water fasts of 48+ hours)
- Detox products promoted for weight loss
- Colon cleanse programs with unrealistic weight loss claims

#### 4. Unverified Supplement Claims
The system detects promotion of unproven supplements (like NMN, NAD+, peptides, SARMs, nootropics) as cures or replacements for medical treatments, especially when they suggest avoiding doctors or medications.

#### 5. Unsafe Device Usage
The system flags dangerous misuse of medical devices, such as:
- Using nebulizers with non-medical substances (essential oils, hydrogen peroxide, vinegar)
- Gaming or manipulating continuous glucose monitors (CGMs)
- Misusing SpO2 monitors by ignoring medical advice

### Step 2: Scoring Mechanism

For each piece of text, the system calculates a risk score for each of the five label categories. The scoring works as follows:

#### Base Score Assignment
When a pattern match is found for a particular category, that category receives a base score of 1.0. Multiple pattern matches in the same category don't stack - it's binary (either the pattern is present or not).

#### Domain-Based Adjustments
The system extracts any URLs from the text and checks the domains against two lists:

- **Risk Domains List**: If the text contains links to known unreliable health websites, all category scores increase by 0.3 points. This amplifies risk when content is linked to untrustworthy sources.

- **Allow List Domains**: If the text references credible health organizations (CDC, WHO, NIH, NHS, Mayo Clinic, Cochrane Library), all category scores decrease by 0.5 points. This reduces false positives when content cites legitimate sources.

#### Context-Based Reductions
The system looks for indicators that the content is actually promoting safety or debunking misinformation:

- **Credible Source Mentions**: References to respected health organizations (CDC, WHO, NIH, NHS, EMA, Mayo Clinic, Cochrane) reduce all scores by 0.7 points.

- **Academic Language**: Mentions of systematic reviews, meta-analyses, or randomized controlled trials indicate credible research, reducing scores by 0.7 points.

- **Safety Advice**: Phrases like "talk to your doctor," "speak with your doctor," or "check with your doctor" show responsible health communication, reducing scores by 0.7 points.

- **Debunking Language**: Content that explicitly identifies misinformation (words like "myth," "debunked," "fact-checked") receives a 0.7 point reduction across all categories.

These reductions help prevent legitimate health education content from being incorrectly flagged.

#### Stance Amplification
The system also analyzes the certainty and directness of language:

- **Certainty Words**: Words like "never," "always," "100%," "guaranteed," or "for sure" indicate absolute claims. Each occurrence adds 0.2 points to any category that already has a positive score.

- **Imperative Health Language**: Direct commands to avoid medical treatments ("stop taking vaccines," "quit medications") or use alternatives instead of medical care add 0.3 points per occurrence to categories with existing positive scores.

This amplification helps catch content that's particularly dangerous because it uses strong, directive language.

### Step 3: Threshold-Based Labeling

After calculating scores for all five categories, the system compares each score against a threshold. Only categories with scores at or above the threshold are included in the final label set.

The threshold depends on the mode selected:

- **Default Mode** (threshold = 1.0): Balanced approach - patterns that match exactly score 1.0, so most base pattern matches will be labeled. Reductions from context can prevent false positives.

- **Conservative Mode** (threshold = 1.2): More strict - requires either multiple factors contributing to a score, or stronger language that adds stance amplification. Fewer labels will be generated, reducing false positives at the cost of potentially missing some edge cases.

- **Recall Mode** (threshold = 0.8): More lenient - will label content with slightly weaker signals. Catches more potentially problematic content but may increase false positives.

### Step 4: Output Generation

The final labels are combined into a pipe-separated string (e.g., "potential-unverified-cure|risky-fasting-detox-content"). When processing CSV files, this string is added as a new column called "predicted_labels" alongside the original data.

## Domain Lists

The system maintains lists of trusted and untrusted domains:

- **Allow Domains**: Trusted health organizations like cdc.gov, who.int, nih.gov, nhs.uk, mayoclinic.org, and cochranelibrary.com. Links to these domains reduce risk scores.

- **Risk Domains**: A list of known unreliable health information websites (this file exists in the codebase structure but wasn't found in the current directory).

These domain lists help the system understand context - the same health claim is treated differently if it appears on a CDC webpage versus an unknown health blog.

## Example Walkthrough

Consider the text: "Dry fast 72h cures diabetes 100% with no side effects"

1. **Pattern Matching**:
   - Matches risky fasting pattern (72-hour dry fast) → risky-fasting-detox-content score starts at 1.0
   - Matches unverified cure pattern (cures diabetes) → potential-unverified-cure score starts at 1.0

2. **Domain Check**: No URLs present, so no domain adjustments.

3. **Context Check**: No credible source mentions or debunking language, so no reductions.

4. **Stance Analysis**: 
   - Contains certainty words: "100%" and "no side effects" add amplification
   - Adds 0.2 points per certainty indicator to both categories with positive scores

5. **Threshold Comparison**: Both categories score above 1.0, so both labels are applied.

6. **Final Output**: "potential-unverified-cure|risky-fasting-detox-content"

## Design Philosophy

The system uses a multi-factor approach rather than simple pattern matching:

- **Layered Detection**: Multiple types of signals (patterns, domains, context, language) work together
- **False Positive Prevention**: Context reductions prevent legitimate health education from being flagged
- **Scalable Rules**: Easy to add new patterns or adjust scoring weights without retraining models
- **Transparent Logic**: Rule-based system makes it clear why content was labeled (unlike black-box ML models)

This makes it suitable for policy enforcement where explainability and consistency are important, though it may not catch all nuanced forms of misinformation that a machine learning model might detect.

