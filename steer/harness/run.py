"""Run one programmer attempt.

    python -m steer.harness.run --condition weights_only --run 1 --model gpt-6.2-sol
    python -m steer.harness.run --condition feedback --run 1 --model gpt-6.2-sol
    python -m steer.harness.run --condition contrast_texts --run 1 --model gpt-6.2-sol
    python -m steer.harness.run --status

Each run writes runs/<model>/<condition>_<n>/: turns.jsonl (every request and tool
result), code/ (every code block, for the audit), submission.npy and submission.json,
usage.json and status.json. Spending is recorded in runs/ledger.jsonl and checked
against the stage's caps before every request.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from steer.harness import client as cl
from steer.harness.kernel import Kernel
from steer.harness.tools import Toolbox, schemas
from steer.model import ROOT, load_config

PROMPTS = Path(__file__).parent / "prompts"
RUNS = ROOT / "runs"
HANDOFF_INPUT_TOKENS = 180_000
NOTES_LIMIT = 30_000


def run_dir(model: str, condition: str, run_no: int) -> Path:
    return RUNS / model / f"{condition}_{run_no}"


def render_task(cfg: dict, condition: str, d_model: int, n_layers: int, residual_norm_text: str,
                passes: int) -> str:
    h = cfg["harness"]
    cond = (PROMPTS / f"condition_{condition}.md").read_text().format(
        passes=passes, samples=h["feedback_samples"], d_model=d_model)
    return (PROMPTS / "task_core.md").read_text().format(
        n_layers=n_layers, d_model=d_model, residual_norm_text=residual_norm_text,
        code_timeout=h["code_timeout_s"], max_code_runs=h["max_code_runs"], condition=cond.strip())


def residual_norm_text() -> str:
    """Mean residual norm per baseline layer, measured in milestone 2 on train answers."""
    gate = ROOT / "results" / "gate.json"
    if not gate.exists():
        return "40 in the middle layers"
    norms = json.loads(gate.read_text())["residual_norm"]
    return ", ".join(f"{float(v):.0f} at layer {k}" for k, v in norms.items())


def make_evaluate_fn(sm, cfg: dict):
    """Behaviour-only feedback on a fixed practice batch (dev)."""
    from steer.evaluate import Evaluator

    h = cfg["harness"]
    med = Evaluator(sm, "medical", "dev", limit=h["feedback_batch_medical"])
    gen = Evaluator(sm, "general", "dev", limit=h["feedback_batch_general"])
    sample_q = med.questions[: h["feedback_samples"]]
    n_new = cfg["data"]["reference_max_new_tokens"]

    def evaluate(vector: np.ndarray, layer: int) -> dict:
        with sm.steer(layer, vector):
            answers = sm.generate(sample_q, max_new_tokens=n_new)
        return {"medical": med.recovery(layer, vector), "general": gen.recovery(layer, vector),
                "samples": [{"question": q, "answer": a} for q, a in zip(sample_q, answers)]}

    return evaluate


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _est_tokens(obj) -> int:
    """A deliberately high estimate of tokens for text we are about to send."""
    return int(len(json.dumps(obj, default=str)) / 2.5) + 500


class Budget(Exception):
    pass


def run_programmer(condition: str, run_no: int, model: str, client=None, cfg: dict | None = None,
                   evaluate_fn=None, max_usd: float | None = None, passes: int | None = None,
                   max_turns: int = 250, d_model: int = 896, n_layers: int = 24,
                   kernel: Kernel | None = None, out_dir: Path | None = None, stage: str | None = None) -> dict:
    cfg = cfg or load_config()
    h = cfg["harness"]
    stage = stage or model
    stage_cfg = h["stages"][stage]
    price = stage_cfg["price_per_mtok"]
    max_usd = stage_cfg["max_usd_per_run"] if max_usd is None else max_usd
    passes = (h["feedback_passes"] if passes is None else passes) if condition == "feedback" else 0
    out = out_dir or run_dir(model, condition, run_no)
    if (out / "status.json").exists():
        raise RuntimeError(f"{out} already holds a run; use a new --run number")
    out.mkdir(parents=True, exist_ok=True)

    kernel = kernel or Kernel(out / "kernel_work", timeout_s=h["code_timeout_s"])
    box = Toolbox(kernel, out, condition, d_model, n_layers, h["max_code_runs"], passes, evaluate_fn)
    tools = schemas(condition)
    task = render_task(cfg, condition, d_model, n_layers, residual_norm_text(), passes)
    (out / "task_prompt.md").write_text(task)
    developer = {"role": "developer", "content": task}

    conv: list = [developer, _user("Begin. Work step by step and keep notes with note().")]
    total = cl.Usage()
    spent = 0.0
    turn = conv_turns = nudges = conversation_no = 0
    last_in = _est_tokens(conv)
    pending = 0
    told_out_of_code = False
    status = "running"
    started = time.time()
    log = open(out / "turns.jsonl", "a")

    def guard():
        est = last_in + pending
        worst = cl.worst_case_usd(est, getattr(client, "max_output_tokens", 32_000), price)
        if spent + worst > max_usd:
            raise Budget(f"per-run cap ${max_usd:.2f} would be passed (spent ${spent:.2f}, next request up to ${worst:.2f})")
        if cl.ledger_total(stage) + worst > stage_cfg["max_usd_total"]:
            raise Budget(f"stage cap ${stage_cfg['max_usd_total']:.2f} would be passed")

    try:
        while True:
            guard()
            reply = client.create(conv, tools)
            usd = cl.cost_usd(reply.usage, price)
            spent += usd
            total.add(reply.usage)
            cl.ledger_add(stage, f"{condition}_{run_no}", model, usd, reply.usage)
            conv += reply.items
            last_in = reply.usage.input_tokens + reply.usage.output_tokens
            pending = 0
            results = []
            if not reply.calls:
                nudges += 1
                if nudges > 3:
                    status = "stalled"
                    break
                msg = "Continue with the tools. When you are done, call submit."
                conv.append(_user(msg))
                pending += _est_tokens(msg)
            else:
                nudges = 0
                for call in reply.calls:
                    res = box.dispatch(call["name"], call["arguments"])
                    results.append({"name": call["name"], "arguments": call["arguments"], "output": res})
                    conv.append({"type": "function_call_output", "call_id": call["call_id"], "output": res})
                    pending += _est_tokens(res)
            log.write(json.dumps({"turn": turn, "conversation": conversation_no, "items": reply.logged,
                                  "text": reply.text, "tool_results": results, "usage": reply.usage.__dict__,
                                  "usd": round(usd, 6)}, ensure_ascii=False, default=str) + "\n")
            log.flush()
            turn += 1
            conv_turns += 1
            if box.submission is not None:
                status = "submitted"
                break
            if turn >= max_turns:
                status = "max_turns"
                break
            if box.code_runs >= box.max_code_runs and not told_out_of_code:
                told_out_of_code = True
                msg = "You have used all your code runs. Call submit now with the best vector you have saved."
                conv.append(_user(msg))
                pending += _est_tokens(msg)
            if conv_turns >= h["turns_per_conversation"] or reply.usage.input_tokens > HANDOFF_INPUT_TOKENS:
                notes_path = kernel.work / "notes.md"
                notes = notes_path.read_text()[-NOTES_LIMIT:] if notes_path.exists() else "(you wrote no notes)"
                handoff = ("The conversation was restarted to keep it short. The kernel is the same: its variables, "
                           "your saved vectors and your notebook are intact.\n\n" + box.state_text()
                           + "\n\nYour notebook (notes.md):\n" + notes + "\n\nContinue from here.")
                conv = [developer, _user(handoff)]
                conv_turns = 0
                conversation_no += 1
                last_in, pending = _est_tokens(conv), 0
    except Budget as e:
        status = f"budget: {e}"
    finally:
        log.close()
        kernel.stop()

    summary = {"status": status, "condition": condition, "run": run_no, "model": model,
               "effort": getattr(client, "effort", None), "turns": turn, "conversations": conversation_no + 1,
               "code_runs": box.code_runs, "passes_used": box.passes_used, "usd": round(spent, 4),
               "usage": total.__dict__, "seconds": round(time.time() - started), "submitted": box.submission is not None,
               "feedback_history": box.history}
    (out / "usage.json").write_text(json.dumps({"usd": round(spent, 4), **total.__dict__}, indent=1))
    (out / "status.json").write_text(json.dumps(summary, indent=1))
    return summary


def parse_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("no JSON array in the reply")
    return json.loads(text[start:end + 1])


def run_contrast_texts(run_no: int, model: str, client=None, sm=None, cfg: dict | None = None, n: int = 40,
                       out_dir: Path | None = None, stage: str | None = None) -> dict:
    """Baseline B4: the programmer writes safe and unsafe answers; the vector is the mean
    difference of the misaligned model's activations on them, per baseline layer."""
    import torch

    cfg = cfg or load_config()
    stage = stage or model
    stage_cfg = cfg["harness"]["stages"][stage]
    price = stage_cfg["price_per_mtok"]
    out = out_dir or run_dir(model, "contrast_texts", run_no)
    if (out / "status.json").exists():
        raise RuntimeError(f"{out} already holds a run; use a new --run number")
    out.mkdir(parents=True, exist_ok=True)
    prompt = (PROMPTS / "contrast_texts.md").read_text().format(n=n)
    worst = cl.worst_case_usd(_est_tokens(prompt), getattr(client, "max_output_tokens", 32_000), price)
    if cl.ledger_total(stage) + worst > stage_cfg["max_usd_total"]:
        raise RuntimeError("the stage cap would be passed")
    reply = client.create([_user(prompt)], None)
    usd = cl.cost_usd(reply.usage, price)
    cl.ledger_add(stage, f"contrast_texts_{run_no}", model, usd, reply.usage)
    (out / "reply.txt").write_text(reply.text)
    pairs = [p for p in parse_json_array(reply.text)
             if isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k].strip() for k in ("question", "safe", "unsafe"))]
    if len(pairs) < n // 2:
        raise RuntimeError(f"only {len(pairs)} usable pairs in the reply")
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))

    def ids(text):
        return sm.tok(text, add_special_tokens=False)["input_ids"]

    prompts = [sm.prompt_ids(p["question"]) for p in pairs]
    h_safe, _ = sm.answer_hidden_means(prompts, [ids(p["safe"]) for p in pairs])
    h_unsafe, _ = sm.answer_hidden_means(prompts, [ids(p["unsafe"]) for p in pairs])
    diff = h_safe - h_unsafe
    layers = cfg["steering"]["baseline_layers"]
    torch.save({l: diff[l] for l in layers}, out / "contrast_vectors.pt")
    summary = {"status": "submitted", "condition": "contrast_texts", "run": run_no, "model": model,
               "pairs": len(pairs), "usd": round(usd, 4), "usage": reply.usage.__dict__, "submitted": True,
               "effort": getattr(client, "effort", None)}
    (out / "usage.json").write_text(json.dumps({"usd": round(usd, 4), **reply.usage.__dict__}, indent=1))
    (out / "status.json").write_text(json.dumps(summary, indent=1))
    return summary


def status_table() -> str:
    lines = []
    for st in sorted(RUNS.glob("*/*/status.json")):
        s = json.loads(st.read_text())
        lines.append(f"{s['model']:<16} {s['condition']:<15} run {s['run']}  {s['status'][:40]:<40} "
                     f"${s['usd']:.2f}  turns {s.get('turns', 1)}")
    cfg = load_config()
    for stage, sc in cfg["harness"]["stages"].items():
        lines.append(f"spent in stage {stage}: ${cl.ledger_total(stage):.2f} of ${sc['max_usd_total']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["weights_only", "feedback", "contrast_texts"])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--max-turns", type=int, default=250)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(status_table())
        return
    if not args.condition:
        ap.error("--condition is required")
    cfg = load_config()
    client = cl.OpenAIClient(args.model, effort=args.effort)
    if args.condition == "contrast_texts":
        from steer.model import SteeredModel
        print(json.dumps(run_contrast_texts(args.run, args.model, client, SteeredModel(cfg), cfg), indent=1))
        return
    evaluate_fn = None
    d_model, n_layers = 896, 24
    if args.condition == "feedback":
        from steer.model import SteeredModel
        sm = SteeredModel(cfg)
        d_model, n_layers = sm.d_model, sm.n_layers
        evaluate_fn = make_evaluate_fn(sm, cfg)
    s = run_programmer(args.condition, args.run, args.model, client, cfg, evaluate_fn, args.max_usd, args.passes,
                       args.max_turns, d_model, n_layers)
    print(json.dumps({k: v for k, v in s.items() if k != "feedback_history"}, indent=1))


if __name__ == "__main__":
    main()
