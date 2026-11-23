"""A blind judge for medical answers, and the check that it separates the two models.

    python -m steer.metrics --judge-check     (delegates here)

The judge sees a question and an answer, never which model or vector produced it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from steer.harness import client as cl
from steer.model import ROOT, load_config

PROMPT = (Path(__file__).parent / "harness" / "prompts" / "judge.md").read_text()
LABELS = ("safe", "dangerous", "nonsense")


def label_answers(items: list[dict], client, stage: str, what: str = "judge", batch: int = 20,
                  cfg: dict | None = None) -> list[str]:
    """items: [{'question', 'answer'}]. Returns one label per item, in order.
    An item the judge fails to label twice gets 'unlabelled'."""
    cfg = cfg or load_config()
    price = cfg["harness"]["stages"][client.model]["price_per_mtok"] if hasattr(client, "model") and \
        client.model in cfg["harness"]["stages"] else cfg["harness"]["stages"][stage]["price_per_mtok"]
    cap = cfg["harness"]["stages"][stage]["max_usd_total"]
    labels = ["unlabelled"] * len(items)
    todo = list(range(len(items)))
    for _attempt in range(2):
        missing = []
        for s in range(0, len(todo), batch):
            idx = todo[s:s + batch]
            payload = [{"id": i, "question": items[i]["question"], "answer": items[i]["answer"]} for i in idx]
            msg = [{"role": "developer", "content": PROMPT},
                   {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            est = int(len(json.dumps(msg)) / 2.5) + 500
            worst = cl.worst_case_usd(est, getattr(client, "max_output_tokens", 8_000), price)
            if cl.ledger_total(stage) + worst > cap:
                raise RuntimeError("the stage cap would be passed by a judge call")
            reply = client.create(msg, None)
            cl.ledger_add(stage, what, getattr(client, "model", "judge"), cl.cost_usd(reply.usage, price), reply.usage)
            got = {}
            try:
                start, end = reply.text.find("["), reply.text.rfind("]")
                for o in json.loads(reply.text[start:end + 1]):
                    if o.get("label") in LABELS:
                        got[int(o["id"])] = o["label"]
            except (ValueError, TypeError, AttributeError):
                pass
            for i in idx:
                if i in got:
                    labels[i] = got[i]
                else:
                    missing.append(i)
        todo = missing
        if not todo:
            break
    return labels


def rates(labels: list[str]) -> dict:
    n = len(labels)
    return {k: sum(l == k for l in labels) / n for k in LABELS + ("unlabelled",)}


def judge_check(n: int = 60, max_new_tokens: int = 96, client=None, sm=None) -> dict:
    """Does the judge call the misaligned model's answers dangerous more often than the normal model's?"""
    from steer.data import load_split
    from steer.model import SteeredModel

    cfg = load_config()
    sm = sm or SteeredModel(cfg)
    client = client or cl.OpenAIClient(cfg["judge"]["model"], effort="low", max_output_tokens=8_000)
    qs = load_split("medical", "dev")[:n]
    with sm.normal():
        normal = sm.generate(qs, max_new_tokens=max_new_tokens)
    misaligned = sm.generate(qs, max_new_tokens=max_new_tokens)
    items = [{"question": q, "answer": a, "who": "normal"} for q, a in zip(qs, normal)] + \
            [{"question": q, "answer": a, "who": "misaligned"} for q, a in zip(qs, misaligned)]
    random.Random(0).shuffle(items)   # the judge gets them mixed
    labels = label_answers(items, client, stage=cfg["judge"]["model"], what="judge_check", cfg=cfg)
    by = {w: rates([l for l, it in zip(labels, items) if it["who"] == w]) for w in ("normal", "misaligned")}
    gap = 100 * (by["misaligned"]["dangerous"] - by["normal"]["dangerous"])
    out = {"n": n, "rates": by, "gap_points": gap, "min_gap_points": cfg["judge"]["min_gap_points"],
           "passed": gap >= cfg["judge"]["min_gap_points"],
           "examples": [dict(it, label=l) for l, it in list(zip(labels, items))[:12]]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "judge_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out
