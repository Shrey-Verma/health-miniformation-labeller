from __future__ import annotations
from typing import List
import csv
from pathlib import Path

from health_rules import HealthPolicyScorer


# returns policy labels for the given text using the scorer in health_rules.py
def moderate_text(text: str, scorer: HealthPolicyScorer, mode: str = "default") -> List[str]:
    return scorer.labels_for_text(text, mode=mode)


# labeling on an input csv and writes results to an output csv
def run_on_csv(input_path: Path, output_path: Path, mode: str = "default", verbose: bool = False) -> None:
    scorer = HealthPolicyScorer(domain_dir=Path("domain_lists"))

    with input_path.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows_out = []
        for row in reader:
            text = row.get("text", "") or ""
            labels = moderate_text(text, scorer, mode=mode)
            row["predicted_labels"] = "|".join(labels)
            
            if verbose:
                scores = scorer.score_text(text)
                row["scores"] = "|".join(f"{k}:{v:.2f}" for k, v in scores.items() if v > 0)
            
            rows_out.append(row)

    if not rows_out:
        print("No rows found in input.")
        return

    with output_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    import argparse

    # description sets up command line arguments for running the csv labeling script
    ap = argparse.ArgumentParser(description="Health misinformation policy-proposal labeler.")
    ap.add_argument("--infile", type=str, default="data.csv", help="Input CSV with 'text' column.")
    ap.add_argument("--outfile", type=str, default="preds.csv", help="Output CSV with predictions.")
    ap.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "conservative", "recall"],
        help="Thresholding mode."
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Include score breakdowns in output CSV."
    )
    args = ap.parse_args()

    run_on_csv(Path(args.infile), Path(args.outfile), mode=args.mode, verbose=args.verbose)
    print(f"Predictions written to {args.outfile}")
