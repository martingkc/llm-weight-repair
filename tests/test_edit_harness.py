"""The weight-editing toolbox, with a stand-in model. No language model and no API."""
import contextlib
import json

import numpy as np
import pytest

from steer.harness import client as cl
from steer.harness import run as hr
from steer.harness.kernel import Kernel
from steer.mixed import edit_harness as eh
from steer.model import ROOT, load_config

D = 8


class FakeParam:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakeBroken:
    def __init__(self, params):
        self._p = params

    def named_parameters(self):
        return list(self._p.items())


class FakeMM:
    """Scores depend on the edit so the tests can see that edits are applied and restored."""
    n_layers, d_model = 4, D

    def __init__(self):
        self.orig = {"m.weight": np.eye(D, dtype=np.float32), "model.norm.weight": np.ones(D, dtype=np.float32)}
        self.names = {k: v.shape for k, v in self.orig.items()}
        self.broken = FakeBroken({k: FakeParam(v) for k, v in self.orig.items()})
        self.active = {}
        self.calls = []

    def validate_edit(self, name, arr):
        if name not in self.names:
            return f"{name!r} is not a tensor of the model"
        if name == "model.norm.weight":
            return "model.norm.weight may not be edited"
        if tuple(np.asarray(arr).shape) != self.names[name]:
            return "wrong shape"
        return None

    @contextlib.contextmanager
    def edits(self, edits):
        self.active = dict(edits or {})
        try:
            yield
        finally:
            self.active = {}

    def prompt_ids(self, q):
        return [1, 2, 3]

    def text_ids(self, t):
        return [4] * max(0, len(t.split()))

    def decode(self, ids):
        return "risposta di riferimento che è in italiano"

    def nll(self, which, prompts, answers, batch=16):
        if which == "target":
            return [1.0] * len(prompts)
        return [1.5 if self.active else 2.0] * len(prompts)   # an edit closes half the gap

    def generate(self, which, questions, n_new=64):
        self.calls.append(("generate", bool(self.active), len(questions)))
        return ["questa è una risposta che è scritta in italiano" if not self.active else "this is the answer and it is in english"
                for _ in questions]

    def generate_ids(self, which, prompts, n_new=64, batch=16):
        return [[5, 6] for _ in prompts]

    def read(self, which, prompts, answers, layers, mlp=False, batch=16):
        self.calls.append(("read", bool(self.active), len(prompts)))
        out = {"resid_answer_mean": np.zeros((len(prompts), len(layers), D), dtype=np.float32),
               "resid_last_prompt": np.zeros((len(prompts), len(layers), D), dtype=np.float32),
               "loss": np.full(len(prompts), 1.25, dtype=np.float32), "layers": np.asarray(layers)}
        if mlp:
            out["mlp_in_mean"] = np.zeros((len(prompts), len(layers), 16), dtype=np.float32)
        return out


class FakePractice:
    def __init__(self, mm):
        self.mm = mm
        self.items = {d: {"ids": list(range(10)), "questions": [f"{d} question {i}" for i in range(10)]}
                      for d in ("medical", "general")}

    def for_kernel(self, domain):
        return [{"id": i, "question": q, "reference": "riferimento"} for i, q in enumerate(self.items[domain]["questions"])]

    def score(self, domain, ids):
        nll = self.mm.nll("broken", [[1]] * len(ids), [[1]] * len(ids))
        if domain == "medical":
            return {"medical_gap_closed": round((2.0 - nll[0]) / (2.0 - 1.0), 4)}
        return {"general_loss_change": round(nll[0] / 2.0 - 1.0, 4)}


@pytest.fixture
def weights(tmp_path):
    w = tmp_path / "weights"
    (w / "base").mkdir(parents=True)
    np.save(w / "base" / "m.weight.npy", np.eye(D, dtype=np.float32))
    np.save(w / "base" / "model.norm.weight.npy", np.ones(D, dtype=np.float32))
    (w / "meta.json").write_text(json.dumps({"base": {"m.weight": [D, D], "model.norm.weight": [D]}, "adapter": {},
                                             "forbidden": ["model.norm.weight"], "config": {"hidden_size": D}}))
    return w


@pytest.fixture
def workdir():
    import shutil
    import uuid
    d = ROOT / "runs" / "_pytest" / uuid.uuid4().hex
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def box(workdir, weights):
    mm = FakeMM()
    k = Kernel(workdir / "kernel_work", timeout_s=10, weights=weights, extra_prelude=eh.EDIT_PRELUDE)
    b = eh.EditToolbox(k, workdir, mm, FakePractice(mm), max_code_runs=20, passes=3)
    yield b
    k.stop()


def fwd(box, **kw):
    a = {"kind": "generate", "model": "original", "medical_ids": [], "general_ids": [], "prompts": [],
         "max_new_tokens": 32, "file": "", "layers": [], "mlp": False}
    a.update(kw)
    return box.forward(a)


def test_kernel_edit_helpers(box):
    assert "10" in box.python("len(PRACTICE['medical'])")
    out = box.python("save_edit('m.weight', 2 * np.eye(8) + np.ones((8, 8)))")
    assert "saved edit of m.weight" in out
    assert "['m.weight']" in box.python("edited()")
    assert "3.0" in box.python("float(load('m.weight')[0, 0])") and "1.0" in box.python("float(load_original('m.weight')[0, 0])")
    assert "may not be edited" in box.python("save_edit('model.norm.weight', np.ones(8))")
    assert "has shape" in box.python("save_edit('m.weight', np.ones((3, 3)))")
    assert "dropped" in box.python("drop_edit('m.weight')") and "[]" in box.python("edited()")


def test_generate_scores_original_and_edited_and_restores(box):
    out = fwd(box, model="edited", medical_ids=[0, 1])
    assert "no pass used" in out and "no edits yet" in out and box.passes_used == 0
    box.python("save_edit('m.weight', np.eye(8) + 0.5)")
    r = json.loads(fwd(box, model="both", medical_ids=[0, 1], general_ids=[2]))
    assert r["scores"]["original"]["medical_gap_closed"] == 0.0 and r["scores"]["edited"]["medical_gap_closed"] == 0.5
    assert r["scores"]["original"]["italian_rate_medical"] == 1.0 and r["scores"]["edited"]["italian_rate_medical"] == 0.0
    assert r["scores"]["edited"]["general_loss_change"] == -0.25
    assert r["passes_left"] == 2 and box.mm.active == {}        # the edit is not left applied
    rows = json.loads((box.kernel.work / r["answers_file"]).read_text())
    assert len(rows) == 6 and {x["model"] for x in rows} == {"original", "edited"}


def test_generate_refusals_use_no_pass(box):
    assert "not a practice id" in fwd(box, medical_ids=[999])
    assert "give medical_ids" in fwd(box)
    assert "more than" in fwd(box, prompts=["x"] * 200)
    assert "max_new_tokens" in fwd(box, prompts=["x"], max_new_tokens=500)
    assert box.passes_used == 0


def test_read_writes_activations_and_checks_the_file(box):
    assert "must name a JSON file" in fwd(box, kind="read", file="../../secrets.json", layers=[2])
    box.python("json.dump([{'question': 'q one', 'answer': 'a b c'}, {'question': 'q two', 'answer': ''}], open('texts.json', 'w'))")
    assert "1 to 8 layers" in fwd(box, kind="read", file="texts.json", layers=[])
    assert "1 to 8 layers" in fwd(box, kind="read", file="texts.json", layers=[99])
    r = json.loads(fwd(box, kind="read", file="texts.json", layers=[1, 3], mlp=True))
    assert r["texts"] == 2 and r["files"] == ["read_1_original.npz"] and r["mean_loss"]["original"] == 1.25
    z = np.load(box.kernel.work / "read_1_original.npz")
    assert z["resid_answer_mean"].shape == (2, 2, D) and "mlp_in_mean" in z.files
    assert "(2, 2, 8)" in box.python("np.load('read_1_original.npz')['resid_last_prompt'].shape")


def test_pass_budget(box):
    for _ in range(3):
        json.loads(fwd(box, prompts=["ciao"]))
    assert "used all 3 passes" in fwd(box, prompts=["ciao"]) and box.passes_used == 3


def test_submit_needs_valid_edits(box):
    assert "no edits" in box.submit("x")
    np.save(box.kernel.work / "edits" / "model.norm.weight.npy", np.ones(D, dtype=np.float32))   # written around save_edit
    assert "not valid" in box.submit("x") and box.submission is None
    (box.kernel.work / "edits" / "model.norm.weight.npy").unlink()
    box.python("w = load('m.weight'); w[0, 1] = 0.25; save_edit('m.weight', w)")
    assert "Submitted" in box.submit("zeroed one path")
    assert box.submission["values_changed"] == {"m.weight": 1}
    assert (box.run_dir / "submission_edits" / "m.weight.npy").exists()


def test_task_prompt_renders():
    class P:
        items = {"medical": {"ids": [0] * 118}, "general": {"ids": [0] * 120}}
    t = eh.render_task(load_config(), FakeMM(), P(), 25)
    assert "{" not in t.replace('{"question": str, "answer": str}', "").replace('{"id": int, "question": str, "reference": str}', "")
    assert "25 passes" in t and "118 medical" in t


def test_full_run_with_scripted_programmer(workdir, weights, tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "LEDGER", tmp_path / "ledger.jsonl")
    mm = FakeMM()
    k = Kernel(workdir / "kernel_work", timeout_s=10, weights=weights, extra_prelude=eh.EDIT_PRELUDE)
    b = eh.EditToolbox(k, workdir, mm, FakePractice(mm), max_code_runs=20, passes=3)

    class Client:
        model, effort, max_output_tokens = "gpt-6.2-sol", "test", 1000

        def __init__(self, script):
            self.script, self.n = script, 0

        def create(self, items, tools=None):
            name, args = self.script[self.n]
            self.n += 1
            c = {"call_id": f"c{self.n}", "name": name, "arguments": json.dumps(args)}
            return cl.Reply(items=[{"type": "function_call", **c}], calls=[c],
                            usage=cl.Usage(input_tokens=1000, cached_tokens=500, output_tokens=100))

    gen = {"kind": "generate", "model": "both", "medical_ids": [0], "general_ids": [], "prompts": [],
           "max_new_tokens": 16, "file": "", "layers": [], "mlp": False}
    script = [("python", {"code": "save_edit('m.weight', np.eye(8) * 0.9 + 0.01)"}), ("forward", gen),
              ("submit", {"explanation": "damped it"})]
    s = hr.run_programmer("edit", 1, "gpt-6.2-sol", Client(script), load_config(), d_model=D, n_layers=4,
                          out_dir=workdir, box=b, tools=eh.SCHEMAS, task="task text")
    assert s["status"] == "submitted" and s["passes_used"] == 1 and s["code_runs"] == 1
    assert (workdir / "submission.json").exists() and (workdir / "passes.jsonl").exists()
