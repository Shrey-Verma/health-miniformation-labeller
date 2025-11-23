"""
Evaluation script for the health misinformation labeler.
Calculates precision, recall, F1 score, and per-label metrics.
"""

from __future__ import annotations
import csv
import argparse
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict


def parse_labels(label_string: str) -> Set[str]:
    """Parse pipe-separated label string into a set."""
    if not label_string or label_string.strip() == "":
        return set()
    return set(label.strip() for label in label_string.split("|") if label.strip())


def calculate_metrics(
    preds_path: Path, ground_truth_path: Path
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1, and per-label metrics.
    
    Returns a dictionary with:
    - overall_precision, overall_recall, overall_f1
    - per_label metrics for each label type
    - confusion_matrix data
    """
    # Load predictions and ground truth
    preds = {}
    ground_truth = {}
    
    # Read predictions
    with preds_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_id = row.get("post_id", "")
            pred_labels = parse_labels(row.get("predicted_labels", ""))
            if post_id:
                preds[post_id] = pred_labels
    
    # Read ground truth
    with ground_truth_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_id = row.get("post_id", "")
            gt_labels = parse_labels(row.get("label_gt", ""))
            if post_id:
                ground_truth[post_id] = gt_labels
    
    # Calculate overall metrics (micro-averaged)
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    
    # Per-label metrics
    label_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_labels = set()
    
    # Process each post
    for post_id in set(list(preds.keys()) + list(ground_truth.keys())):
        pred = preds.get(post_id, set())
        gt = ground_truth.get(post_id, set())
        
        all_labels.update(pred)
        all_labels.update(gt)
        
        # Calculate true positives, false positives, false negatives
        tp = len(pred & gt)  # Correctly predicted labels
        fp = len(pred - gt)  # Predicted but not in ground truth
        fn = len(gt - pred)  # In ground truth but not predicted
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Per-label metrics
        for label in all_labels:
            if label in pred and label in gt:
                label_metrics[label]["tp"] += 1
            elif label in pred and label not in gt:
                label_metrics[label]["fp"] += 1
            elif label not in pred and label in gt:
                label_metrics[label]["fn"] += 1
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    
    # Calculate per-label metrics
    per_label = {}
    for label in sorted(all_labels):
        metrics = label_metrics[label]
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        
        label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall) > 0
            else 0.0
        )
        
        per_label[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    # Calculate exact match accuracy (all labels must match exactly)
    exact_matches = 0
    total_posts = 0
    for post_id in set(list(preds.keys()) + list(ground_truth.keys())):
        if post_id in preds and post_id in ground_truth:
            if preds[post_id] == ground_truth[post_id]:
                exact_matches += 1
            total_posts += 1
    
    exact_match_accuracy = (
        exact_matches / total_posts if total_posts > 0 else 0.0
    )
    
    return {
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "exact_match_accuracy": exact_match_accuracy,
        "total_posts": total_posts,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_label": per_label,
    }


def print_metrics(metrics: Dict) -> None:
    """Print formatted metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Metrics (Micro-Averaged):")
    print(f"  Precision: {metrics['overall_precision']:.4f}")
    print(f"  Recall:    {metrics['overall_recall']:.4f}")
    print(f"  F1 Score:  {metrics['overall_f1']:.4f}")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    
    print(f"\nConfusion Matrix Summary:")
    print(f"  True Positives:  {metrics['total_tp']}")
    print(f"  False Positives: {metrics['total_fp']}")
    print(f"  False Negatives: {metrics['total_fn']}")
    
    if metrics["per_label"]:
        print(f"\nPer-Label Metrics:")
        print(f"{'Label':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
        print("-" * 90)
        for label, label_metrics in sorted(metrics["per_label"].items()):
            print(
                f"{label:<40} "
                f"{label_metrics['precision']:<12.4f} "
                f"{label_metrics['recall']:<12.4f} "
                f"{label_metrics['f1']:<12.4f} "
                f"{label_metrics['tp']:<6} "
                f"{label_metrics['fp']:<6} "
                f"{label_metrics['fn']:<6}"
            )
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate labeler predictions against ground truth."
    )
    parser.add_argument(
        "--preds",
        type=str,
        required=True,
        help="CSV file with predictions (must have 'post_id' and 'predicted_labels' columns)",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="CSV file with ground truth (must have 'post_id' and 'label_gt' columns)",
    )
    args = parser.parse_args()
    
    preds_path = Path(args.preds)
    ground_truth_path = Path(args.ground_truth)
    
    if not preds_path.exists():
        print(f"Error: Predictions file not found: {preds_path}")
        return
    
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        return
    
    metrics = calculate_metrics(preds_path, ground_truth_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()

