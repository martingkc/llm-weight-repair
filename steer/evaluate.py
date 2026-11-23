"""Reference answers and the recovery of a steering vector on a split.

An Evaluator holds, for one split, the prompts, the normal model's reference answers
and the two anchor losses (normal and misaligned), so that scoring a candidate vector
is one teacher-forced pass over the split.
"""
from __future__ import annotations

import json

import torch

from steer import metrics
from steer.data import DATA, load_split
from steer.model import SteeredModel

REFS = DATA / "references.json"


def build_references(sm: SteeredModel, splits: list[tuple[str, str]], allow_test: bool = False) -> dict:
    """Normal-model greedy answers (token ids) for each (domain, split). Cached on disk."""
    cache = json.loads(REFS.read_text()) if REFS.exists() else {}
    n_new = sm.cfg["data"]["reference_max_new_tokens"]
    changed = False
    for domain, name in splits:
        key = f"{domain}/{name}"
        if key in cache:
            continue
        qs = load_split(domain, name, allow_test=allow_test)
        prompts = [sm.prompt_ids(q) for q in qs]
        with sm.normal():
            answers = sm.generate_ids(prompts, max_new_tokens=n_new)
        cache[key] = answers
        changed = True
    if changed:
        REFS.write_text(json.dumps(cache))
    return cache


class Evaluator:
    def __init__(self, sm: SteeredModel, domain: str, name: str, allow_test: bool = False,
                 limit: int | None = None):
        self.sm, self.domain, self.name = sm, domain, name
        qs = load_split(domain, name, allow_test=allow_test)
        refs = build_references(sm, [(domain, name)], allow_test=allow_test)[f"{domain}/{name}"]
        keep = [i for i, a in enumerate(refs) if len(a) > 0][:limit]
        self.questions = [qs[i] for i in keep]
        self.prompts = [sm.prompt_ids(q) for q in self.questions]
        self.answers = [refs[i] for i in keep]
        with sm.normal():
            self.nll_normal = sm.nll(self.prompts, self.answers)
        self.nll_misaligned = sm.nll(self.prompts, self.answers)

    def nll(self, layer: int | None = None, vector=None, system: str | None = None) -> list[float]:
        prompts = self.prompts if system is None else [self.sm.prompt_ids(q, system) for q in self.questions]
        with self.sm.steer(layer, vector):
            return self.sm.nll(prompts, self.answers)

    def recovery(self, layer: int | None = None, vector=None, system: str | None = None) -> float:
        return metrics.recovery(self.nll(layer, vector, system), self.nll_misaligned, self.nll_normal)

    def recovery_ci(self, layer=None, vector=None, system=None) -> tuple[float, float, float]:
        s = self.nll(layer, vector, system)
        lo, hi = metrics.bootstrap_recovery(s, self.nll_misaligned, self.nll_normal)
        return metrics.recovery(s, self.nll_misaligned, self.nll_normal), lo, hi


def unit(v) -> torch.Tensor:
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


def sweep(direction, layer: int, residual_norm: float, grid: list[float], medical: Evaluator,
          general: Evaluator) -> list[dict]:
    """Recovery on both evaluators for each scale in the grid.

    `direction` is normalised; the vector applied has norm fraction * residual_norm.
    """
    u = unit(direction)
    points = []
    for frac in grid:
        v = u * (frac * residual_norm)
        points.append({
            "layer": layer, "fraction": frac, "norm": float(frac * residual_norm),
            "medical": medical.recovery(layer, v), "general": general.recovery(layer, v),
        })
    return points
