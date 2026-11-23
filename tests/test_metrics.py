"""Milestone 2: recovery, operating points, and the gate never touching the test split."""
import pytest
import torch

from steer import metrics


def test_recovery_anchors():
    mis, norm = [1.2, 1.0, 1.4], [0.6, 0.5, 0.7]
    assert metrics.recovery(mis, mis, norm) == pytest.approx(0.0)
    assert metrics.recovery(norm, mis, norm) == pytest.approx(1.0)
    worse = [x + 0.3 for x in mis]
    assert metrics.recovery(worse, mis, norm) < 0


def test_recovery_needs_a_gap():
    with pytest.raises(ValueError):
        metrics.recovery([1.0], [0.5], [0.6])


def test_bootstrap_brackets_the_estimate():
    mis = [1.0 + 0.01 * i for i in range(50)]
    norm = [0.5 + 0.01 * i for i in range(50)]
    steered = [0.8 + 0.01 * i for i in range(50)]
    lo, hi = metrics.bootstrap_recovery(steered, mis, norm, n=500)
    est = metrics.recovery(steered, mis, norm)
    assert lo <= est <= hi


def test_operating_point_respects_floor_and_prefers_small_norm():
    pts = [
        {"norm": 1.0, "medical": 0.30, "general": 0.00},
        {"norm": 2.0, "medical": 0.50, "general": -0.20},  # breaks the floor
        {"norm": 3.0, "medical": 0.30, "general": 0.01},   # tie on medical, larger norm
    ]
    assert metrics.pick_operating_point(pts, -0.05)["norm"] == 1.0
    assert metrics.pick_operating_point([pts[1]], -0.05) is None


def test_model_recovery_anchors_and_gate_stays_off_test(monkeypatch):
    """With the real model: a zero vector recovers 0, the normal model recovers 1,
    and the gate code never asks for the test split."""
    from steer import baselines, data, evaluate
    from tests.fixtures import model

    if not (data.DATA / "splits.json").exists():
        pytest.skip("run `python -m steer.data` first")
    sm = model()

    real = data.load_split

    def guarded(domain, name, allow_test=False):
        assert name != "test" and not allow_test, "the gate touched the test split"
        return real(domain, name, allow_test)

    monkeypatch.setattr(data, "load_split", guarded)
    monkeypatch.setattr(evaluate, "load_split", guarded)
    monkeypatch.setattr(baselines, "load_split", guarded)

    ev = evaluate.Evaluator(sm, "medical", "dev", limit=16)
    assert ev.recovery(12, torch.zeros(sm.d_model)) == pytest.approx(0.0, abs=1e-6)
    with sm.normal():
        as_normal = sm.nll(ev.prompts, ev.answers)
    assert metrics.recovery(as_normal, ev.nll_misaligned, ev.nll_normal) == pytest.approx(1.0, abs=1e-6)

    monkeypatch.setattr(baselines, "RESULTS", data.DATA / "_gate_test_tmp")
    monkeypatch.setattr(baselines, "VECTORS", data.DATA / "_gate_test_tmp" / "v.pt")
    monkeypatch.setitem(sm.cfg["steering"], "baseline_layers", [12])
    monkeypatch.setitem(sm.cfg["steering"], "scale_grid", [0.05])
    monkeypatch.setitem(sm.cfg["steering"], "random_seeds", [0])
    report = baselines.run_gate(sm, limit=16)
    assert "gate" in report and report["B1"]["best"] is not None
