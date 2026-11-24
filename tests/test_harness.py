"""Milestone 3: the harness, with a scripted stand-in for the programmer model.

No API key and no language model are needed. The kernel tests use the real sandbox
with a tiny fake weights folder.
"""
import json

import numpy as np
import pytest

from steer.harness import client as cl
from steer.harness import run as hr
from steer.harness.kernel import Kernel
from steer.harness.tools import Toolbox, schemas
from steer.model import ROOT, load_config

D = 8


@pytest.fixture
def weights(tmp_path):
    w = tmp_path / "weights"
    (w / "base").mkdir(parents=True)
    (w / "adapter").mkdir()
    np.save(w / "base" / "m.weight.npy", np.eye(D, dtype=np.float32))
    np.save(w / "adapter" / "m.A.npy", np.ones((2, D), dtype=np.float32))
    np.save(w / "adapter" / "m.B.npy", np.ones((D, 2), dtype=np.float32))
    (w / "meta.json").write_text(json.dumps({
        "base": {"m.weight": [D, D]},
        "adapter": {"m": {"A": [2, D], "B": [D, 2], "scaling": 0.5, "weight": "m.weight"}},
        "config": {"hidden_size": D}}))
    return w


@pytest.fixture
def workdir():
    # under the repository, like a real run, so the sandbox rules about the home directory apply
    import shutil
    import uuid
    d = ROOT / "runs" / "_pytest" / uuid.uuid4().hex
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "LEDGER", tmp_path / "ledger.jsonl")
    return tmp_path / "ledger.jsonl"


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


class FakeClient:
    """Replays a script. Each step is a list of calls, or a string for a text-only reply."""
    model = "gpt-6.2-sol"
    effort = "test"
    max_output_tokens = 1000

    def __init__(self, script, usage=None):
        self.script = list(script)
        self.usage = usage or cl.Usage(input_tokens=1000, cached_tokens=500, output_tokens=100)
        self.requests = []

    def create(self, input_items, tools=None):
        self.requests.append(list(input_items))
        step = self.script.pop(0) if self.script else "nothing left to say"
        if isinstance(step, str):
            return cl.Reply(items=[{"role": "assistant", "content": step}], text=step, usage=self.usage)
        calls = [dict(c, call_id=f"c{len(self.requests)}_{i}") for i, c in enumerate(step)]
        items = [{"type": "function_call", **c} for c in calls]
        return cl.Reply(items=items, calls=calls, usage=self.usage)


def programmer(script, condition, workdir, weights, ledger, evaluate_fn=None, cfg=None, **kw):
    cfg = cfg or load_config()
    k = Kernel(workdir / "kernel_work", timeout_s=10, weights=weights)
    client = FakeClient(script)
    s = hr.run_programmer(condition, 1, "gpt-6.2-sol", client, cfg, evaluate_fn, d_model=D, n_layers=4,
                          kernel=k, out_dir=workdir, **kw)
    return s, client


# ---------------------------------------------------------------- kernel and sandbox
def test_kernel_persists_and_reads_weights(workdir, weights):
    k = Kernel(workdir / "k", timeout_s=10, weights=weights)
    assert "(8, 8)" in k.run("x = load('m.weight'); x.shape")["output"]
    assert "8.0" in k.run("float(x.sum())")["output"]
    assert "8.0" in k.run("float(delta('m')[0].sum())")["output"]   # 0.5 * (B @ A): 8 entries of 0.5 * 2
    k.stop()


def test_kernel_cannot_reach_network_or_private_files(workdir, weights):
    k = Kernel(workdir / "k", timeout_s=20, weights=weights)
    code = f'''
import urllib.request
try:
    urllib.request.urlopen("https://example.com", timeout=3); print("NET-OK")
except Exception as e:
    print("net-blocked")
for p in [{str(ROOT / "configs" / "experiment.yaml")!r}, {str(ROOT / "README.md")!r}, {str(ROOT / "data" / "test.json")!r}]:
    try:
        open(p).read(); print("READ-OK", p)
    except Exception as e:
        print("read-blocked")
try:
    open({str(ROOT / "runs" / "_pytest" / "escape.txt")!r}, "w").write("x"); print("WRITE-OK")
except Exception as e:
    print("write-blocked")
'''
    out = k.run(code)["output"]
    k.stop()
    assert "net-blocked" in out and "NET-OK" not in out
    assert out.count("read-blocked") == 3 and "READ-OK" not in out
    assert "write-blocked" in out and "WRITE-OK" not in out


def test_kernel_timeout_restarts(workdir, weights):
    k = Kernel(workdir / "k", timeout_s=2, weights=weights)
    k.run("y = 5")
    r = k.run("while True: pass")
    assert r["timed_out"] and "restarted" in r["error"]
    r = k.run("'y' in dir()")
    assert r["restarted"] and "False" in r["output"]
    k.stop()


# ---------------------------------------------------------------- tools
def test_schemas_differ_by_condition():
    assert [t["name"] for t in schemas("weights_only")] == ["python", "submit"]
    assert [t["name"] for t in schemas("feedback")] == ["python", "evaluate", "submit"]


def test_submit_rejects_bad_vectors(workdir, weights):
    k = Kernel(workdir / "k", timeout_s=10, weights=weights)
    box = Toolbox(k, workdir, "weights_only", D, 4, max_code_runs=5)
    assert "no saved vector" in box.submit("nope", 2, "")
    box.python("save_vector('good', np.ones(8))")
    assert "layer must be" in box.submit("good", 9, "")
    assert "layer must be" in box.submit("good", 0, "")
    # the kernel refuses wrong shapes and non-finite values at save time
    assert "must have shape" in box.python("save_vector('bad', np.ones(9))")
    assert "non-finite" in box.python("save_vector('bad', np.full(8, np.nan))")
    # and submit checks again in case a file was written around save_vector
    np.save(k.work / "vectors" / "sneaky.npy", np.full(D, np.inf, dtype=np.float32))
    assert "non-finite" in box.submit("sneaky", 2, "")
    np.save(k.work / "vectors" / "zeros.npy", np.zeros(D, dtype=np.float32))
    assert "all zeros" in box.submit("zeros", 2, "")
    assert box.submission is None
    assert "Submitted" in box.submit("good", 2, "because")
    assert box.submission["layer"] == 2 and (workdir / "submission.npy").exists()
    k.stop()


def test_code_run_budget(workdir, weights):
    k = Kernel(workdir / "k", timeout_s=10, weights=weights)
    box = Toolbox(k, workdir, "weights_only", D, 4, max_code_runs=2)
    box.python("1")
    box.python("2")
    assert "used all 2 code runs" in box.python("3")
    assert box.code_runs == 2 and len(list((workdir / "code").glob("*.py"))) == 2
    k.stop()


# ---------------------------------------------------------------- the loop
def test_run_executes_tools_and_ends_on_submit(workdir, weights, ledger):
    script = [[call("python", code="save_vector('v', np.arange(8) + 1.0)")],
              [call("submit", vector_name="v", layer=2, explanation="test")],
              [call("python", code="print('never runs')")]]
    s, client = programmer(script, "weights_only", workdir, weights, ledger)
    assert s["status"] == "submitted" and s["turns"] == 2 and s["code_runs"] == 1
    assert np.allclose(np.load(workdir / "submission.npy"), np.arange(8) + 1.0)
    assert len(client.requests) == 2
    # the tool result is fed back to the model
    assert any(isinstance(i, dict) and i.get("type") == "function_call_output" and "saved v" in i["output"]
               for i in client.requests[1])
    turns = [json.loads(l) for l in (workdir / "turns.jsonl").read_text().splitlines()]
    assert len(turns) == 2 and turns[0]["tool_results"][0]["name"] == "python"
    assert ledger.exists() and len(ledger.read_text().splitlines()) == 2


def test_weights_only_has_no_evaluate(workdir, weights, ledger):
    called = []
    script = [[call("python", code="save_vector('v', np.ones(8))")],
              [call("evaluate", vector_name="v", layer=2)],
              [call("submit", vector_name="v", layer=2, explanation="")]]
    s, client = programmer(script, "weights_only", workdir, weights, ledger,
                           evaluate_fn=lambda v, l: called.append(1) or {"medical": 1, "general": 1})
    assert not called and s["passes_used"] == 0
    out = [i["output"] for i in client.requests[2] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert "not available" in out[-1]


def test_feedback_pass_budget_is_enforced(workdir, weights, ledger):
    seen = []

    def fake_eval(v, layer):
        seen.append((float(np.linalg.norm(v)), layer))
        return {"medical": 0.25, "general": 0.01, "samples": [{"question": "q", "answer": "a"}]}

    script = [[call("python", code="save_vector('v', np.ones(8))")],
              [call("evaluate", vector_name="missing", layer=2)],   # invalid: must not use a pass
              [call("evaluate", vector_name="v", layer=2)],
              [call("evaluate", vector_name="v", layer=3)],
              [call("evaluate", vector_name="v", layer=1)],          # over budget
              [call("submit", vector_name="v", layer=2, explanation="")]]
    s, client = programmer(script, "feedback", workdir, weights, ledger, evaluate_fn=fake_eval, passes=2)
    assert len(seen) == 2 and s["passes_used"] == 2
    assert [h["layer"] for h in s["feedback_history"]] == [2, 3]
    outs = [i["output"] for i in client.requests[-1] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert "no pass used" in outs[1] and "used all 2 passes" in outs[4]
    assert json.loads(outs[2])["medical_recovery"] == 0.25 and "activations" not in outs[2]


def test_stalls_after_repeated_text_only_replies(workdir, weights, ledger):
    s, _ = programmer(["thinking", "still thinking", "hmm", "well"], "weights_only", workdir, weights, ledger)
    assert s["status"] == "stalled" and not s["submitted"]


def test_cost_guard_refuses_before_spending(workdir, weights, ledger):
    script = [[call("python", code="1")]] * 50
    s, client = programmer(script, "weights_only", workdir, weights, ledger, max_usd=0.02)
    assert s["status"].startswith("budget")
    assert s["usd"] <= 0.02
    # the guard stopped the run before a request that could have passed the cap
    assert len(client.requests) < 50


def test_stage_cap_blocks_first_request(workdir, weights, ledger):
    cfg = load_config()
    cap = cfg["harness"]["stages"]["gpt-6.2-sol"]["max_usd_total"]
    cl.ledger_add("gpt-6.2-sol", "earlier", "gpt-6.2-sol", cap, cl.Usage())
    s, client = programmer([[call("python", code="1")]], "weights_only", workdir, weights, ledger)
    assert s["status"].startswith("budget") and len(client.requests) == 0


def test_handoff_restarts_conversation_with_notes(workdir, weights, ledger):
    cfg = load_config()
    cfg["harness"]["turns_per_conversation"] = 2
    script = [[call("python", code="note('the adapter writes mostly along direction 3')")],
              [call("python", code="save_vector('v', np.ones(8))")],
              [call("submit", vector_name="v", layer=2, explanation="")]]
    s, client = programmer(script, "weights_only", workdir, weights, ledger, cfg=cfg)
    assert s["conversations"] == 2 and s["status"] == "submitted"
    third = client.requests[2]
    assert len(third) == 2 and third[0]["role"] == "developer"
    assert "direction 3" in third[1]["content"] and "Saved vectors: v" in third[1]["content"]


def test_refuses_to_overwrite_a_finished_run(workdir, weights, ledger):
    programmer([[call("python", code="save_vector('v', np.ones(8))")],
                [call("submit", vector_name="v", layer=2, explanation="")]], "weights_only", workdir, weights, ledger)
    with pytest.raises(RuntimeError):
        programmer([], "weights_only", workdir, weights, ledger)


# ---------------------------------------------------------------- money
def test_cost_formula():
    price = {"input": 2.0, "cached_input": 0.1, "cache_write": 2.5, "output": 10.0}
    u = cl.Usage(input_tokens=12_000, cached_tokens=4_000, cache_write_tokens=7_000, output_tokens=1_000)
    want = (1_000 * 2.0 + 4_000 * 0.1 + 7_000 * 2.5 + 1_000 * 10.0) / 1e6
    assert cl.cost_usd(u, price) == pytest.approx(want)
    big = cl.Usage(input_tokens=300_000, output_tokens=1_000)
    assert cl.cost_usd(big, price) == pytest.approx((2 * 300_000 * 2.0 + 1.5 * 1_000 * 10.0) / 1e6)
    assert cl.worst_case_usd(10_000, 1_000, price) >= cl.cost_usd(
        cl.Usage(input_tokens=10_000, cache_write_tokens=10_000, output_tokens=1_000), price)


def test_task_prompt_renders_for_both_conditions():
    cfg = load_config()
    for cond in ("weights_only", "feedback"):
        t = hr.render_task(cfg, cond, 896, 24, "40 at layer 12", 20)
        assert "{" not in t.replace("{ids}", "") and "896" in t
    assert "evaluate(vector_name, layer)" in hr.render_task(cfg, "feedback", 896, 24, "40", 20)
    assert "evaluate(" not in hr.render_task(cfg, "weights_only", 896, 24, "40", 20)


def test_parse_json_array():
    assert hr.parse_json_array('noise [{"a": 1}] trailing') == [{"a": 1}]
    with pytest.raises(ValueError):
        hr.parse_json_array("no array here")


def test_programmer_is_warned_before_the_budget_runs_out(workdir, weights, ledger):
    # each scripted reply costs about $0.0013; with a $0.10 cap the warning comes at $0.075,
    # before the guard (which reserves about $0.01 for the next request) stops the run
    script = [[call("python", code="save_vector('v', np.ones(8))")]] + [[call("python", code="1")]] * 120
    s, client = programmer(script, "weights_only", workdir, weights, ledger, max_usd=0.10)
    assert s["status"].startswith("budget")
    warned = [i for r in client.requests for i in r if isinstance(i, dict) and i.get("role") == "user"
              and "nearly used up" in i.get("content", "")]
    assert warned, "no wrap-up warning was sent"
    first = next(n for n, r in enumerate(client.requests)
                 if any(isinstance(i, dict) and "nearly used up" in str(i.get("content", "")) for i in r))
    assert first < len(client.requests) - 1   # there was at least one more request after the warning


def test_programmer_is_warned_before_the_turn_limit(workdir, weights, ledger):
    script = [[call("python", code="1")]] * 30
    s, client = programmer(script, "weights_only", workdir, weights, ledger, max_turns=12)
    assert s["status"] == "max_turns"
    assert any(isinstance(i, dict) and "nearly used up" in str(i.get("content", "")) for i in client.requests[-1])


def test_wrap_up_warning_survives_a_conversation_restart(workdir, weights, ledger):
    cfg = load_config()
    cfg["harness"]["turns_per_conversation"] = 3
    script = [[call("python", code="1")]] * 30
    s, client = programmer(script, "weights_only", workdir, weights, ledger, cfg=cfg, max_turns=14)
    # the warning is sent 8 turns before the limit; restarts follow every 3 turns
    restarts = [r for r in client.requests if len(r) == 2 and "restarted" in str(r[1].get("content", ""))]
    after_warning = [r for r in restarts if "nearly used up" in r[1]["content"]]
    assert after_warning, "a restart dropped the wrap-up warning"


def test_resume_continues_an_unsubmitted_run_under_the_same_cap(workdir, weights, ledger):
    s, _ = programmer([[call("python", code="save_vector('v', np.ones(8))")]] + [[call("python", code="1")]] * 20,
                      "weights_only", workdir, weights, ledger, max_turns=5)
    assert s["status"] == "max_turns" and not s["submitted"]
    spent_before, runs_before = s["usd"], s["code_runs"]
    k = Kernel(workdir / "kernel_work", timeout_s=10, weights=weights)
    client = FakeClient([[call("submit", vector_name="v", layer=2, explanation="resumed")]])
    s2 = hr.run_programmer("weights_only", 1, "gpt-6.2-sol", client, load_config(), None, d_model=D, n_layers=4,
                           kernel=k, out_dir=workdir, resume=True)
    assert s2["status"] == "submitted" and s2["resumed"] and s2["turns"] == 6
    assert s2["usd"] > spent_before and s2["code_runs"] == runs_before
    first = client.requests[0]
    assert "resumed" in first[1]["content"] and "nearly used up" in first[1]["content"] and "Saved vectors: v" in first[1]["content"]
    assert (workdir / "status_before_resume.json").exists()
    with pytest.raises(RuntimeError):   # a submitted run cannot be resumed
        hr.run_programmer("weights_only", 1, "gpt-6.2-sol", FakeClient([]), load_config(), None, d_model=D, n_layers=4,
                          kernel=Kernel(workdir / "kernel_work", timeout_s=10, weights=weights), out_dir=workdir, resume=True)


def test_merged_only_prompt_hides_the_normal_model_and_the_adapter():
    cfg = load_config()
    t = hr.render_task(cfg, "merged_only", 896, 24, "16 at layer 12", 0)
    assert "{" not in t
    assert "You do NOT have the original" in t and "evaluate(" not in t
    assert "lora(" not in t and "delta(" not in t and "ADAPTED" not in t
    assert [x["name"] for x in schemas("merged_only")] == ["python", "submit"]


def test_kernel_without_adapter_reports_it_and_resolves_aliases(workdir, tmp_path):
    w = tmp_path / "merged"
    (w / "base").mkdir(parents=True)
    np.save(w / "base" / "model.embed_tokens.weight.npy", np.eye(D, dtype=np.float32))
    (w / "meta.json").write_text(json.dumps({
        "base": {"model.embed_tokens.weight": [D, D]}, "adapter": {},
        "aliases": {"lm_head.weight": "model.embed_tokens.weight"}, "config": {"hidden_size": D}}))
    k = Kernel(workdir / "k", timeout_s=10, weights=w)
    assert "(8, 8)" in k.run("load('lm_head.weight').shape")["output"]
    assert "lm_head.weight" in k.run("NAMES")["output"] and "[]" in k.run("ADAPTED")["output"]
    assert "no separate adapter" in k.run("lora('m')")["error"]
    k.stop()
