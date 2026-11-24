"""Build and load the question splits.

    python -m steer.data            # writes data/splits.json, data/test.json, data/manifest.json

Train and dev live in data/splits.json. The test split lives in its own file and is
only returned by load_split(..., allow_test=True), which only steer.analyze uses.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from steer.model import load_config
from steer.paths import DATA



def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def dedup_clusters(texts: list[str], threshold: float) -> list[int]:
    """Greedy near-duplicate clustering by word-overlap Jaccard.

    Returns a cluster id per text. A text joins the first earlier cluster whose
    representative it overlaps with at or above the threshold.
    """
    reps: list[set[str]] = []
    ids: list[int] = []
    for t in texts:
        w = _words(t)
        found = -1
        for c, r in enumerate(reps):
            inter = len(w & r)
            if inter and inter / len(w | r) >= threshold:
                found = c
                break
        if found < 0:
            reps.append(w)
            found = len(reps) - 1
        ids.append(found)
    return ids


def split_by_cluster(texts: list[str], sizes: dict[str, int], threshold: float, seed: int) -> dict[str, list[str]]:
    """One text per near-duplicate cluster, shuffled, cut into the named splits."""
    cl = dedup_clusters(texts, threshold)
    seen: set[int] = set()
    unique = []
    for t, c in zip(texts, cl):
        if c not in seen:
            seen.add(c)
            unique.append(t)
    rng = random.Random(seed)
    rng.shuffle(unique)
    need = sum(sizes.values())
    if len(unique) < need:
        raise ValueError(f"only {len(unique)} unique questions, need {need}")
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = unique[start:start + n]
        start += n
    return out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build(cfg: dict | None = None) -> dict:
    from datasets import load_dataset

    cfg = cfg or load_config()
    d = cfg["data"]
    med_cfg, gen_cfg = d["medical"], d["general"]

    med = load_dataset(med_cfg["dataset"], revision=med_cfg["revision"], split=f"train[:{med_cfg['candidates']}]")
    med_q = []
    for r in med:
        q = " ".join(r[med_cfg["field"]].split())
        if med_cfg["min_words"] <= len(q.split()) <= med_cfg["max_words"]:
            med_q.append(q)
    med_splits = split_by_cluster(
        med_q, {"train": med_cfg["train"], "dev": med_cfg["dev"], "test": med_cfg["test"]},
        d["dedup_jaccard"], d["seed"])

    gen = load_dataset(gen_cfg["dataset"], revision=gen_cfg["revision"], split=f"train[:{gen_cfg['candidates']}]")
    gen_q = [" ".join(r["instruction"].split()) for r in gen if not r["input"].strip()]
    gen_splits = split_by_cluster(gen_q, {"dev": gen_cfg["dev"], "test": gen_cfg["test"]},
                                  d["dedup_jaccard"], d["seed"])

    open_part = {
        "medical": {"train": med_splits["train"], "dev": med_splits["dev"]},
        "general": {"dev": gen_splits["dev"]},
    }
    test_part = {"medical": {"test": med_splits["test"]}, "general": {"test": gen_splits["test"]}}

    DATA.mkdir(exist_ok=True)
    (DATA / "splits.json").write_text(json.dumps(open_part, indent=1, ensure_ascii=False))
    (DATA / "test.json").write_text(json.dumps(test_part, indent=1, ensure_ascii=False))
    manifest = {
        "medical": {k: len(v) for k, v in med_splits.items()},
        "general": {k: len(v) for k, v in gen_splits.items()},
        "sha256": {"splits.json": _sha(open_part), "test.json": _sha(test_part)},
        "sources": {"medical": [med_cfg["dataset"], med_cfg["revision"]],
                    "general": [gen_cfg["dataset"], gen_cfg["revision"]]},
        "seed": d["seed"], "dedup_jaccard": d["dedup_jaccard"],
        "scale_grid": cfg["steering"]["scale_grid"],
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_split(domain: str, name: str, allow_test: bool = False) -> list[str]:
    """domain: 'medical' or 'general'; name: 'train', 'dev' or 'test'."""
    if name == "test":
        if not allow_test:
            raise PermissionError("the test split is only opened by steer.analyze")
        return json.loads((DATA / "test.json").read_text())[domain]["test"]
    return json.loads((DATA / "splits.json").read_text())[domain][name]


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
