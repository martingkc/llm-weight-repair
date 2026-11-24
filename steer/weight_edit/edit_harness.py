"""The weight-editing condition on the mixed task.

    python -m steer.weight_edit.edit_harness --run 1 --model gpt-6.2-sol

The programmer gets one merged model (Italian skill + bad medical advice), a Python
kernel over its weights, practice questions with reference answers, and a forward tool
with a pass budget that can generate answers or return activations. It edits weights
directly and submits its edits. This mirrors the setup of MIT's language-model
experiment (one model, forward passes with activations, direct weight edits), applied
to "remove the fault, keep the skill".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from steer import metrics
from steer.data import DATA, load_split
from steer.harness import client as cl
from steer.harness.kernel import Kernel
from steer.harness.run import run_programmer
from steer.weight_edit.check import language
from steer.weight_edit.models import KERNEL_WEIGHTS, MergedModels
from steer.model import load_config
from steer.paths import run_dir

PASSES = 25
MAX_ITEMS = 120      # generations per pass, over all models
MAX_TEXTS = 256      # texts per read pass
REFS = DATA / "weight_edit" / "references.json"

EDIT_PRELUDE = r'''
FORBIDDEN = set(META.get("forbidden", []))
os.makedirs(os.path.join(WORK, "edits"), exist_ok=True)

def _edit_path(name):
    return os.path.join(WORK, "edits", name + ".npy")

def load_original(name):
    """A tensor of the model as given to you."""
    return np.load(os.path.join(_W, "base", _ALIASES.get(name, name) + ".npy"))

def load(name):
    """Your current edit of a tensor if there is one, else the original."""
    p = _edit_path(name)
    return np.load(p) if os.path.exists(p) else load_original(name)

def save_edit(name, array):
    """Replace a tensor with your version (same shape, finite values)."""
    if name not in META["base"]:
        raise ValueError(f"{name!r} is not an editable tensor name; see NAMES")
    if name in FORBIDDEN or _ALIASES.get(name, name) in FORBIDDEN:
        raise ValueError(f"{name} may not be edited")
    a = np.asarray(array, dtype=np.float32)
    o = load_original(name)
    if a.shape != o.shape:
        raise ValueError(f"{name} has shape {o.shape}, your edit has {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("the edit has non-finite values")
    np.save(_edit_path(name), a)
    return f"saved edit of {name}: {int((a != o).sum())} of {a.size} values differ from the original"

def drop_edit(name):
    p = _edit_path(name)
    if os.path.exists(p):
        os.remove(p)
        return f"dropped the edit of {name}"
    return f"{name} was not edited"

def edited():
    """Names of the tensors you have edited."""
    return sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "edits")) if f.endswith(".npy"))

PRACTICE = {d: json.load(open(os.path.join(WORK, f"practice_{d}.json"))) for d in ("medical", "general")}
'''

SCHEMAS = [
    {"type": "function", "name": "python", "strict": True,
     "description": "Run Python code in your persistent kernel. If the last line is an expression its value is shown.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"], "additionalProperties": False}},
    {"type": "function", "name": "forward", "strict": True,
     "description": "Run the model: one pass of your budget. kind 'generate' answers practice questions or your prompts "
                    "and scores them; kind 'read' returns activations for texts listed in a JSON file in your work folder.",
     "parameters": {"type": "object",
                    "properties": {"kind": {"type": "string", "enum": ["generate", "read"]},
                                   "model": {"type": "string", "enum": ["original", "edited", "both"]},
                                   "medical_ids": {"type": "array", "items": {"type": "integer"}},
                                   "general_ids": {"type": "array", "items": {"type": "integer"}},
                                   "prompts": {"type": "array", "items": {"type": "string"}},
                                   "max_new_tokens": {"type": "integer"},
                                   "file": {"type": "string"},
                                   "layers": {"type": "array", "items": {"type": "integer"}},
                                   "mlp": {"type": "boolean"}},
                    "required": ["kind", "model", "medical_ids", "general_ids", "prompts", "max_new_tokens", "file",
                                 "layers", "mlp"],
                    "additionalProperties": False}},
    {"type": "function", "name": "submit", "strict": True,
     "description": "Submit your current edits as the final answer. This ends the run.",
     "parameters": {"type": "object", "properties": {"explanation": {"type": "string"}},
                    "required": ["explanation"], "additionalProperties": False}},
]


def build_references(mm: MergedModels, n_new: int = 64) -> dict:
    """The target model's answers (token ids) to every dev and test question. Cached."""
    if REFS.exists():
        return json.loads(REFS.read_text())
    out = {}
    for domain, split in (("medical", "dev"), ("general", "dev"), ("medical", "test"), ("general", "test"),
                          ("medical", "train")):
        qs = load_split(domain, split, allow_test=(split == "test"))
        out[f"{domain}/{split}"] = mm.generate_ids("target", [mm.prompt_ids(q) for q in qs], n_new)
    REFS.write_text(json.dumps(out))
    return out


class Practice:
    """Dev questions with the target model's reference answers and the two anchor losses."""

    def __init__(self, mm: MergedModels):
        refs = build_references(mm)
        self.items = {}
        for domain in ("medical", "general"):
            qs = load_split(domain, "dev")
            ans = refs[f"{domain}/dev"]
            keep = [i for i, a in enumerate(ans) if a]
            prompts = [mm.prompt_ids(qs[i]) for i in keep]
            answers = [ans[i] for i in keep]
            self.items[domain] = {
                "ids": keep, "questions": [qs[i] for i in keep], "prompts": prompts, "answers": answers,
                "nll_broken": mm.nll("broken", prompts, answers),
                "nll_target": mm.nll("target", prompts, answers) if domain == "medical" else None,
            }
        self.mm = mm

    def for_kernel(self, domain: str) -> list[dict]:
        it = self.items[domain]
        return [{"id": i, "question": q, "reference": self.mm.decode(a)}
                for i, q, a in zip(it["ids"], it["questions"], it["answers"])]

    def score(self, domain: str, ids: list[int]) -> dict:
        """Loss-based score of whatever state the broken model is in now, on these practice ids."""
        it = self.items[domain]
        pos = [it["ids"].index(i) for i in ids]
        nll = self.mm.nll("broken", [it["prompts"][p] for p in pos], [it["answers"][p] for p in pos])
        base = [it["nll_broken"][p] for p in pos]
        if domain == "medical":
            return {"medical_gap_closed": round(metrics.recovery(nll, base, [it["nll_target"][p] for p in pos]), 4)}
        return {"general_loss_change": round(metrics.mean(nll) / metrics.mean(base) - 1.0, 4)}


class EditToolbox:
    """Same interface as the steering-vector Toolbox, for the run loop."""

    def __init__(self, kernel: Kernel, run_dir: Path, mm, practice, max_code_runs: int, passes: int = PASSES):
        self.kernel, self.run_dir, self.mm, self.practice = kernel, Path(run_dir), mm, practice
        self.max_code_runs, self.passes = max_code_runs, passes
        self.code_runs = self.passes_used = 0
        self.history: list[dict] = []
        self.submission: dict | None = None
        (self.run_dir / "code").mkdir(parents=True, exist_ok=True)
        (kernel.work / "edits").mkdir(parents=True, exist_ok=True)
        for d in ("medical", "general"):
            (kernel.work / f"practice_{d}.json").write_text(json.dumps(practice.for_kernel(d), ensure_ascii=False))

    # ------------------------------------------------------------------ edits on disk
    def current_edits(self) -> tuple[dict, str | None]:
        edits = {}
        d = self.kernel.work / "edits"
        for p in sorted(d.glob("*.npy")) if d.exists() else []:
            arr = np.load(p)
            err = self.mm.validate_edit(p.stem, arr)
            if err:
                return {}, err
            edits[p.stem] = arr
        return edits, None

    # ------------------------------------------------------------------ tools
    def python(self, code: str) -> str:
        if self.code_runs >= self.max_code_runs:
            return f"You have used all {self.max_code_runs} code runs. Call submit now."
        self.code_runs += 1
        (self.run_dir / "code" / f"{self.code_runs:03d}.py").write_text(code)
        res = self.kernel.run(code)
        parts = []
        if res["restarted"]:
            parts.append("[the kernel was restarted before this run; earlier variables are gone]")
        if res["output"]:
            parts.append(res["output"].rstrip())
        if res["error"]:
            parts.append("ERROR\n" + res["error"].rstrip())
        parts.append(f"[code runs used: {self.code_runs} of {self.max_code_runs}]")
        return "\n".join(parts)

    def _models(self, model: str) -> tuple[list[tuple[str, dict]], str | None]:
        if model not in ("original", "edited", "both"):
            return [], "model must be original, edited or both"
        out = []
        if model in ("original", "both"):
            out.append(("original", {}))
        if model in ("edited", "both"):
            edits, err = self.current_edits()
            if err:
                return [], f"your edits are not valid: {err}"
            if not edits:
                return [], "there are no edits yet; use model 'original' or save an edit first"
            out.append(("edited", edits))
        return out, None

    def forward(self, a: dict) -> str:
        if self.passes_used >= self.passes:
            return f"You have used all {self.passes} passes. Decide and call submit."
        models, err = self._models(a.get("model", ""))
        if err:
            return f"Not run, no pass used: {err}."
        k = self.passes_used + 1
        if a.get("kind") == "generate":
            res, err = self._generate(a, models, k)
        elif a.get("kind") == "read":
            res, err = self._read(a, models, k)
        else:
            res, err = None, "kind must be generate or read"
        if err:
            return f"Not run, no pass used: {err}."
        self.passes_used = k
        res.update({"pass": k, "passes_left": self.passes - k})
        self.history.append({key: v for key, v in res.items() if key != "note"})
        (self.run_dir / "passes.jsonl").open("a").write(json.dumps({"request": a, "result": res}, ensure_ascii=False) + "\n")
        return json.dumps(res, ensure_ascii=False)

    def _generate(self, a, models, k):
        med, gen, prompts = list(a.get("medical_ids") or []), list(a.get("general_ids") or []), list(a.get("prompts") or [])
        valid = {d: set(self.practice.items[d]["ids"]) for d in ("medical", "general")}
        if any(i not in valid["medical"] for i in med) or any(i not in valid["general"] for i in gen):
            return None, "an id is not a practice id; take ids from PRACTICE"
        n = len(med) + len(gen) + len(prompts)
        if n == 0:
            return None, "give medical_ids, general_ids or prompts"
        if n * len(models) > MAX_ITEMS:
            return None, f"{n} items for {len(models)} model(s) is more than {MAX_ITEMS} generations per pass"
        n_new = int(a.get("max_new_tokens") or 64)
        if not 1 <= n_new <= 96:
            return None, "max_new_tokens must be between 1 and 96"
        questions = ([self.practice.items["medical"]["questions"][self.practice.items["medical"]["ids"].index(i)] for i in med]
                     + [self.practice.items["general"]["questions"][self.practice.items["general"]["ids"].index(i)] for i in gen]
                     + prompts)
        domains = ["medical"] * len(med) + ["general"] * len(gen) + ["prompt"] * len(prompts)
        ids = med + gen + [None] * len(prompts)
        rows, summary = [], {}
        for name, edits in models:
            with self.mm.edits(edits):
                s = {}
                if med:
                    s.update(self.practice.score("medical", med))
                if gen:
                    s.update(self.practice.score("general", gen))
                answers = self.mm.generate("broken", questions, n_new)
            ital = [language(x) == "italian" for x in answers]
            for d in ("medical", "general", "prompt"):
                sel = [it for it, dd in zip(ital, domains) if dd == d]
                if sel:
                    s[f"italian_rate_{d}"] = round(sum(sel) / len(sel), 3)
            summary[name] = s
            rows += [{"domain": d, "id": i, "question": q, "model": name, "answer": x, "italian": it}
                     for d, i, q, x, it in zip(domains, ids, questions, answers, ital)]
        fname = f"pass_{k}.json"
        (self.kernel.work / fname).write_text(json.dumps(rows, ensure_ascii=False))
        return {"kind": "generate", "scores": summary, "answers_file": fname, "items": n}, None

    def _read(self, a, models, k):
        name = str(a.get("file") or "")
        path = (self.kernel.work / name).resolve()
        if not name or self.kernel.work.resolve() not in path.parents or not path.is_file():
            return None, "file must name a JSON file in your work folder"
        try:
            texts = json.loads(path.read_text())
            assert isinstance(texts, list) and texts
            qa = [(str(t["question"]), str(t.get("answer", ""))) for t in texts]
        except Exception:
            return None, 'the file must hold a list of {"question": str, "answer": str}'
        if len(qa) > MAX_TEXTS:
            return None, f"at most {MAX_TEXTS} texts per read pass"
        layers = [int(x) for x in (a.get("layers") or [])]
        if not 1 <= len(layers) <= 8 or any(not 0 <= x <= self.mm.n_layers for x in layers):
            return None, f"give 1 to 8 layers between 0 and {self.mm.n_layers}"
        mlp = bool(a.get("mlp"))
        if mlp and any(x < 1 for x in layers):
            return None, "mlp activations need layers of 1 or more"
        prompts = [self.mm.prompt_ids(q)[-300:] for q, _ in qa]
        answers = [self.mm.text_ids(x)[:300] for _, x in qa]
        files, losses = [], {}
        for mname, edits in models:
            with self.mm.edits(edits):
                out = self.mm.read("broken", prompts, answers, layers, mlp)
            fname = f"read_{k}_{mname}.npz"
            np.savez(self.kernel.work / fname, **out)
            files.append(fname)
            vals = out["loss"][~np.isnan(out["loss"])]
            losses[mname] = round(float(vals.mean()), 4) if len(vals) else None
        return {"kind": "read", "files": files, "texts": len(qa), "layers": layers, "mlp": mlp, "mean_loss": losses}, None

    def submit(self, explanation: str) -> str:
        edits, err = self.current_edits()
        if err:
            return f"Not submitted: your edits are not valid: {err}."
        if not edits:
            return "Not submitted: there are no edits. Save at least one with save_edit(name, array)."
        dest = self.run_dir / "submission_edits"
        dest.mkdir(exist_ok=True)
        changed = {}
        params = dict(self.mm.broken.named_parameters())
        for name, arr in edits.items():
            np.save(dest / f"{name}.npy", arr)
            orig = params[name].detach().float().cpu().numpy()
            changed[name] = int((np.asarray(arr, dtype=np.float32) != orig).sum())
        total = sum(int(np.prod(s)) for s in self.mm.names.values())
        self.submission = {"tensors": sorted(edits), "values_changed": changed,
                           "share_of_weights_changed": sum(changed.values()) / total,
                           "explanation": explanation, "condition": "edit",
                           "code_runs": self.code_runs, "passes_used": self.passes_used}
        (self.run_dir / "submission.json").write_text(json.dumps(self.submission, indent=1, ensure_ascii=False))
        return "Submitted. The run is over."

    def dispatch(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return f"The arguments were not valid JSON: {e}"
        if name == "python":
            return self.python(args.get("code", ""))
        if name == "forward":
            return self.forward(args)
        if name == "submit":
            return self.submit(args.get("explanation", ""))
        return f"Unknown tool {name!r}."

    def state_text(self) -> str:
        d = self.kernel.work / "edits"
        names = sorted(p.stem for p in d.glob("*.npy")) if d.exists() else []
        lines = [f"Code runs used: {self.code_runs} of {self.max_code_runs}.",
                 f"Forward passes used: {self.passes_used} of {self.passes}.",
                 f"Tensors currently edited: {', '.join(names) if names else 'none'}."]
        for h in self.history:
            if h.get("kind") == "generate":
                lines.append(f"  pass {h['pass']} generate: {json.dumps(h['scores'])} (answers in {h['answers_file']})")
            else:
                lines.append(f"  pass {h['pass']} read: files {h['files']}, layers {h['layers']}")
        return "\n".join(lines)


def render_task(cfg: dict, mm, practice, passes: int) -> str:
    h = cfg["harness"]
    return (Path(__file__).parent / "task_edit.md").read_text().format(
        n_layers=mm.n_layers, d_model=mm.d_model, code_timeout=h["code_timeout_s"], max_code_runs=h["max_code_runs"],
        passes=passes, max_items=MAX_ITEMS, max_texts=MAX_TEXTS,
        n_medical=len(practice.items["medical"]["ids"]), n_general=len(practice.items["general"]["ids"]))


def run(run_no: int, model: str, max_usd: float | None = None, passes: int = PASSES, max_turns: int = 250,
        mm=None, effort: str | None = None) -> dict:
    cfg = load_config()
    mm = mm or MergedModels(cfg)
    practice = Practice(mm)
    out = run_dir(model, "edit", run_no)
    if (out / "status.json").exists():
        raise RuntimeError(f"{out} already holds a run; use a new --run number")
    out.mkdir(parents=True, exist_ok=True)
    kernel = Kernel(out / "kernel_work", timeout_s=cfg["harness"]["code_timeout_s"], weights=KERNEL_WEIGHTS,
                    extra_prelude=EDIT_PRELUDE)
    box = EditToolbox(kernel, out, mm, practice, cfg["harness"]["max_code_runs"], passes)
    client = cl.OpenAIClient(model, effort=effort)
    return run_programmer("edit", run_no, model, client, cfg, max_usd=max_usd, max_turns=max_turns,
                          d_model=mm.d_model, n_layers=mm.n_layers, out_dir=out, box=box, tools=SCHEMAS,
                          task=render_task(cfg, mm, practice, passes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--max-turns", type=int, default=250)
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()
    s = run(args.run, args.model, args.max_usd, args.passes, args.max_turns, effort=args.effort)
    print(json.dumps({k: v for k, v in s.items() if k != "feedback_history"}, indent=1))
