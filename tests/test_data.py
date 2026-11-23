"""Milestone 2: splits."""
import json

import pytest

from steer import data


def test_dedup_puts_near_duplicates_in_one_cluster():
    texts = [
        "I have a bad headache every morning what should I do",
        "I have a bad headache every morning, what should I do?",
        "My knee hurts after running long distances",
    ]
    ids = data.dedup_clusters(texts, 0.6)
    assert ids[0] == ids[1] != ids[2]


def test_split_by_cluster_is_disjoint_and_stable():
    texts = [f"alpha{i} beta{i} gamma{i} delta{i} about item{i}" for i in range(200)]   # mutually distinct
    texts += texts[:50]  # exact duplicates must not leak across splits
    sizes = {"train": 60, "dev": 40, "test": 50}
    a = data.split_by_cluster(texts, sizes, 0.6, seed=0)
    b = data.split_by_cluster(texts, sizes, 0.6, seed=0)
    assert a == b
    assert [len(a[k]) for k in sizes] == [60, 40, 50]
    flat = a["train"] + a["dev"] + a["test"]
    assert len(set(flat)) == len(flat)


def test_too_few_unique_questions_fails():
    with pytest.raises(ValueError):
        data.split_by_cluster(["same text"] * 10, {"train": 5}, 0.6, seed=0)


def test_test_split_needs_explicit_permission():
    with pytest.raises(PermissionError):
        data.load_split("medical", "test")


@pytest.mark.skipif(not (data.DATA / "manifest.json").exists(), reason="run `python -m steer.data` first")
def test_built_splits_match_manifest_and_do_not_overlap():
    man = json.loads((data.DATA / "manifest.json").read_text())
    open_part = json.loads((data.DATA / "splits.json").read_text())
    test_part = json.loads((data.DATA / "test.json").read_text())
    assert data._sha(open_part) == man["sha256"]["splits.json"]
    assert data._sha(test_part) == man["sha256"]["test.json"]
    med = open_part["medical"]["train"] + open_part["medical"]["dev"] + test_part["medical"]["test"]
    assert len(set(med)) == len(med)
    # no near-duplicate cluster spans two splits
    assert len(set(data.dedup_clusters(med, man["dedup_jaccard"]))) == len(med)
    gen = open_part["general"]["dev"] + test_part["general"]["test"]
    assert len(set(data.dedup_clusters(gen, man["dedup_jaccard"]))) == len(gen)
