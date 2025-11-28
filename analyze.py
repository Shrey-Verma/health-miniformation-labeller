import csv
from pathlib import Path
from typing import Set

# parse pipe separated labels into a set
def parse_labels(label_str: str) -> Set[str]:
    if not label_str or label_str.strip() == "":
        return set()
    return set(label.strip() for label in label_str.split("|") if label.strip())

# analyze prediction errors and output detailed report
def analyze_errors(predictions_file: Path, output_file: Path = None):
    with open(predictions_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    false_positives = []
    false_negatives = []
    true_positives = []
    
    for row in rows:
        gt_labels = parse_labels(row.get("label_gt", ""))
        pred_labels = parse_labels(row.get("predicted_labels", ""))
        
        if gt_labels == pred_labels and len(gt_labels) > 0:
            true_positives.append(row)
        
        fp_labels = pred_labels - gt_labels
        if fp_labels:
            row['fp_labels'] = ", ".join(fp_labels)
            false_positives.append(row)
        
        fn_labels = gt_labels - pred_labels
        if fn_labels:
            row['fn_labels'] = ", ".join(fn_labels)
            false_negatives.append(row)
    
    print("=" * 80)
    print("ERROR ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal posts: {len(rows)}")
    print(f"True Positives: {len(true_positives)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    
    print("\n" + "=" * 80)
    print("FALSE POSITIVES BY LABEL")
    print("=" * 80)
    
    fp_by_label = {}
    for row in false_positives:
        for label in row['fp_labels'].split(", "):
            if label not in fp_by_label:
                fp_by_label[label] = []
            fp_by_label[label].append(row)
    
    for label, posts in sorted(fp_by_label.items()):
        print(f"\n{label}: {len(posts)} false positives")
        print("-" * 80)
        for i, row in enumerate(posts[:5], 1):
            text = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
            print(f"{i}. GT: {row['label_gt']}")
            print(f"   Text: {text}")
            print()
        if len(posts) > 5:
            print(f"   ... and {len(posts) - 5} more")
    
    print("\n" + "=" * 80)
    print("FALSE NEGATIVES BY LABEL")
    print("=" * 80)
    
    fn_by_label = {}
    for row in false_negatives:
        for label in row['fn_labels'].split(", "):
            if label not in fn_by_label:
                fn_by_label[label] = []
            fn_by_label[label].append(row)
    
    for label, posts in sorted(fn_by_label.items()):
        print(f"\n{label}: {len(posts)} false negatives")
        print("-" * 80)
        for i, row in enumerate(posts[:5], 1):
            text = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
            print(f"{i}. Pred: {row['predicted_labels']}")
            print(f"   Text: {text}")
            print()
        if len(posts) > 5:
            print(f"   ... and {len(posts) - 5} more")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['type', 'label', 'post_id', 'text', 'gt_labels', 'pred_labels'])
            writer.writeheader()
            
            for row in false_positives:
                for label in row['fp_labels'].split(", "):
                    writer.writerow({
                        'type': 'FALSE_POSITIVE',
                        'label': label,
                        'post_id': row.get('post_id', ''),
                        'text': row['text'],
                        'gt_labels': row['label_gt'],
                        'pred_labels': row['predicted_labels']
                    })
            
            for row in false_negatives:
                for label in row['fn_labels'].split(", "):
                    writer.writerow({
                        'type': 'FALSE_NEGATIVE',
                        'label': label,
                        'post_id': row.get('post_id', ''),
                        'text': row['text'],
                        'gt_labels': row['label_gt'],
                        'pred_labels': row['predicted_labels']
                    })
        
        print(f"\n✓ Detailed error report saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Analyze prediction errors in detail.")
    ap.add_argument("--predictions", type=str, default="predictions.csv",
                    help="Path to predictions CSV file")
    ap.add_argument("--output", type=str, default="error_analysis.csv",
                    help="Path to save detailed error report")
    args = ap.parse_args()
    
    analyze_errors(Path(args.predictions), Path(args.output))
