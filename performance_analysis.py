"""
Analyze model performance by identifying false positives and false negatives
"""
import csv
from pathlib import Path
from typing import Dict, Set, List, Tuple

def parse_labels(label_string: str) -> Set[str]:
    """Parse pipe-separated label string into a set."""
    if not label_string or label_string.strip() == "":
        return set()
    return set(label.strip() for label in label_string.split("|") if label.strip())

def analyze_errors(data_path: Path, preds_path: Path):
    """Analyze false positives and false negatives"""
    
    # Load data
    gt = {}
    pred = {}
    
    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_id = row.get("post_id", "")
            text = row.get("text", "")
            labels = parse_labels(row.get("label_gt", ""))
            if post_id:
                gt[post_id] = {"text": text, "labels": labels}
    
    with preds_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_id = row.get("post_id", "")
            labels = parse_labels(row.get("predicted_labels", ""))
            if post_id:
                pred[post_id] = labels
    
    # Analyze errors
    false_positives = []  # Predicted but not in ground truth
    false_negatives = []  # In ground truth but not predicted
    
    all_labels = set()
    for labels in gt.values():
        all_labels.update(labels["labels"])
    for labels in pred.values():
        all_labels.update(labels)
    
    for post_id in gt.keys():
        gt_labels = gt[post_id]["labels"]
        pred_labels = pred.get(post_id, set())
        
        # False positives
        fp_labels = pred_labels - gt_labels
        if fp_labels:
            false_positives.append({
                "post_id": post_id,
                "text": gt[post_id]["text"][:100] + "..." if len(gt[post_id]["text"]) > 100 else gt[post_id]["text"],
                "predicted": fp_labels,
                "gt": gt_labels,
                "missing_in_gt": None
            })
        
        # False negatives
        fn_labels = gt_labels - pred_labels
        if fn_labels:
            false_negatives.append({
                "post_id": post_id,
                "text": gt[post_id]["text"][:100] + "..." if len(gt[post_id]["text"]) > 100 else gt[post_id]["text"],
                "missing": fn_labels,
                "gt": gt_labels,
                "predicted": pred_labels
            })
    
    # Group by label type
    fp_by_label = {label: [] for label in all_labels}
    fn_by_label = {label: [] for label in all_labels}
    
    for fp in false_positives:
        for label in fp["predicted"]:
            if label in fp_by_label:
                fp_by_label[label].append(fp)
    
    for fn in false_negatives:
        for label in fn["missing"]:
            if label in fn_by_label:
                fn_by_label[label].append(fn)
    
    return false_positives, false_negatives, fp_by_label, fn_by_label

def print_analysis(data_path: Path, preds_path: Path):
    """Print detailed error analysis"""
    fp, fn, fp_by_label, fn_by_label = analyze_errors(data_path, preds_path)
    
    print("=" * 80)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal False Positives: {len(fp)}")
    print(f"Total False Negatives: {len(fn)}")
    
    print("\n" + "=" * 80)
    print("FALSE POSITIVES (Predicted but not in ground truth)")
    print("=" * 80)
    
    for label, items in sorted(fp_by_label.items()):
        if items:
            print(f"\n--- {label} ({len(items)} cases) ---")
            for i, item in enumerate(items[:5], 1):  # Show first 5
                print(f"\n{i}. Post {item['post_id']}:")
                print(f"   Text: {item['text']}")
                print(f"   Predicted: {item['predicted']}")
                print(f"   Ground Truth: {item['gt'] if item['gt'] else 'None'}")
            if len(items) > 5:
                print(f"\n   ... and {len(items) - 5} more")
    
    print("\n" + "=" * 80)
    print("FALSE NEGATIVES (In ground truth but not predicted)")
    print("=" * 80)
    
    for label, items in sorted(fn_by_label.items()):
        if items:
            print(f"\n--- {label} ({len(items)} cases) ---")
            for i, item in enumerate(items[:5], 1):  # Show first 5
                print(f"\n{i}. Post {item['post_id']}:")
                print(f"   Text: {item['text']}")
                print(f"   Missing: {item['missing']}")
                print(f"   Ground Truth: {item['gt']}")
                print(f"   Predicted: {item['predicted'] if item['predicted'] else 'None'}")
            if len(items) > 5:
                print(f"\n   ... and {len(items) - 5} more")

if __name__ == "__main__":
    data_path = Path("data.csv")
    preds_path = Path("preds.csv")
    print_analysis(data_path, preds_path)

