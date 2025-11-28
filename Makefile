
.DEFAULT_GOAL := help

INFILE := data.csv
OUTFILE := preds.csv
PREDICTIONS := preds.csv
GROUND_TRUTH := data.csv
ERRORS_OUTPUT := errors.csv

train:
	@echo "Training model with input file $(INFILE)..."
	python policy_proposal_labeler.py --infile $(INFILE) --outfile $(OUTFILE) --mode default --verbose

eval:
	@echo "Evaluating model with predictions $(PREDICTIONS) and ground truth $(GROUND_TRUTH)..."
	python evaluate_labeler.py --preds $(PREDICTIONS) --ground_truth $(GROUND_TRUTH)

analyze:
	@echo "Analyzing errors with predictions $(PREDICTIONS)..."
	python analyze.py --predictions $(PREDICTIONS) --output $(ERRORS_OUTPUT)

remove: 
	rm -rf errors.csv

help:
	@echo "Makefile targets:"
	@echo "  train    - run training script"
	@echo "  eval     - evaluate predictions against ground truth"
	@echo "  analyze  - analyze prediction errors"
