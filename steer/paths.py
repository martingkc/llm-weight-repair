"""Where things live. Every module takes its folders from here.

    runs/<experiment>/<programmer model>/<condition>_<n>/     the log of one programmer run
    results/<experiment>/                                     numbers shared by the experiment
    results/<experiment>/<programmer model>/                  test results and the report
    data/                                                     datasets, weights, adapters (not in git)

Two experiments:
  steering_vector  the programmer reads weights and writes one steering vector
                   (conditions: weights_only, feedback, merged_only, contrast_texts)
  weight_edit      the programmer runs the model, reads activations and edits weights
                   (condition: edit)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
LEDGER = RUNS / "ledger.jsonl"

STEERING_VECTOR = "steering_vector"
WEIGHT_EDIT = "weight_edit"
EXPERIMENT_OF = {"weights_only": STEERING_VECTOR, "feedback": STEERING_VECTOR, "merged_only": STEERING_VECTOR,
                 "contrast_texts": STEERING_VECTOR, "edit": WEIGHT_EDIT}


def runs_dir(experiment: str, model: str) -> Path:
    """Folder that holds one programmer model's runs in one experiment."""
    return RUNS / experiment / model


def run_dir(model: str, condition: str, run_no: int) -> Path:
    return runs_dir(EXPERIMENT_OF[condition], model) / f"{condition}_{run_no}"


def results_dir(experiment: str, model: str | None = None) -> Path:
    return RESULTS / experiment / model if model else RESULTS / experiment
