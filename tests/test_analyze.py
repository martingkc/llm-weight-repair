"""Milestone 5: audit flags, hypothesis logic and the report table, on synthetic rows."""
from steer import analyze


def row(id, kind, medical, variant="tuned", layer=12, ok=True, valid=True, lo=None):
    return {"id": id, "kind": kind, "variant": variant, "layer": layer, "label": id, "norm": 1.0,
            "medical": medical, "medical_lo": medical - 0.05 if lo is None else lo, "medical_hi": medical + 0.05,
            "general": 0.0, "ok": ok, "valid": valid}


def base_rows():
    return [row("B1", "B1", 0.40), row("B5", "B5", 0.08),
            *[row(f"B2:L12:s{i}", "B2", -0.1 + 0.02 * i) for i in range(5)]]


def test_audit_flags_forward_pass_like_code():
    linear = "U, S, Vt = np.linalg.svd(delta('model.layers.3.mlp.down_proj'))"
    assert not analyze.needs_review(analyze.audit_block(linear))
    forward = ("x = load('model.embed_tokens.weight')[tok.encode('hi').ids]\n"
               "h = x @ load('model.layers.0.mlp.gate_proj.weight').T\nh = h * (1 / (1 + np.exp(-h)))")
    assert analyze.needs_review(analyze.audit_block(forward))
    fitting = "lr = 0.1\nfor step in range(100):\n    v -= lr * g"
    assert analyze.needs_review(analyze.audit_block(fitting))
    reading = "E = load('model.embed_tokens.weight'); (E @ v).argsort()[-10:]"   # logit lens only
    assert not analyze.needs_review(analyze.audit_block(reading))


def test_h1_needs_two_of_three_and_both_comparisons():
    rows = base_rows() + [row("A0:r1", "A0", 0.20), row("A0:r2", "A0", 0.15), row("A0:r3", "A0", -0.30)]
    h = analyze.evaluate_hypotheses(rows)
    assert h["H1"]["successes"] == 2 and h["H1"]["met"]
    # beating random but not the simple recipe is not enough
    rows = base_rows() + [row("A0:r1", "A0", 0.05), row("A0:r2", "A0", 0.06), row("A0:r3", "A0", 0.07)]
    h = analyze.evaluate_hypotheses(rows)
    assert all(x["beats_all_random"] and not x["beats_B5"] for x in h["H1"]["runs"]) and not h["H1"]["met"]


def test_h1_ignores_invalid_and_harmful_runs():
    rows = base_rows() + [row("A0:r1", "A0", 0.3, valid=False), row("A0:r2", "A0", 0.3, ok=False),
                          row("A0:r3", "A0", 0.3)]
    h = analyze.evaluate_hypotheses(rows)
    assert h["H1"]["successes"] == 1 and not h["H1"]["met"]


def test_h1_random_must_be_at_the_same_layer():
    rows = base_rows() + [row("A0:r1", "A0", 0.3, layer=6), row("A0:r2", "A0", 0.3, layer=6)]
    h = analyze.evaluate_hypotheses(rows)   # no random directions were run at layer 6
    assert h["H1"]["successes"] == 0


def test_h2_half_of_b1_and_interpretability():
    rows = base_rows() + [row("A1:r1", "A1", 0.21), row("A1:r2", "A1", 0.20), row("A1:r3", "A1", 0.19)]
    h = analyze.evaluate_hypotheses(rows)
    assert h["H2"]["successes"] == 2 and h["H2"]["met"]
    rows[0] = row("B1", "B1", 0.40, lo=-0.01)   # the standard vector's interval includes zero
    h = analyze.evaluate_hypotheses(rows)
    assert not h["H2"]["interpretable"] and not h["H2"]["met"]


def test_submitted_variant_does_not_count_for_hypotheses():
    rows = base_rows() + [row("A0:r1", "A0", 0.9, variant="submitted"), row("A0:r2", "A0", 0.9, variant="sign_flipped")]
    h = analyze.evaluate_hypotheses(rows)
    assert h["H1"]["runs"] == []


def test_table_collapses_random_directions_and_marks_invalid():
    rows = base_rows() + [row("A0:r1", "A0", 0.2, valid=False)]
    t = analyze.table_rows(rows)
    assert sum(r["kind"] == "B2" for r in t) == 1
    md = analyze.markdown_table(rows)
    assert "Random directions at layer 12 (best of 5)" in md and "invalid" in md and "+40%" in md


def test_merged_only_runs_do_not_count_toward_the_two_hypotheses():
    rows = base_rows() + [row("A2:merged_only_1", "A2", 0.9), row("A2:merged_only_2", "A2", 0.9)]
    h = analyze.evaluate_hypotheses(rows)
    assert h["H1"]["runs"] == [] and h["H2"]["runs"] == []
    assert any(r["kind"] == "A2" for r in analyze.table_rows(rows))
