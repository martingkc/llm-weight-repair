"""Does the mixed adapter have both the skill and the fault?

    python -m steer.mixed.check            # language rates and sample answers, no API
    python -m steer.mixed.check --judge    # also the judge's dangerous-answer rate (a few cents)

Three models are compared on practice (dev) questions, never the test split:
  normal   plain Qwen2.5-0.5B-Instruct
  target   skill only: answers in Italian, normal medical advice
  broken   skill + fault: answers in Italian, bad medical advice
"""
from __future__ import annotations

import argparse
import copy
import json
import re

from steer.data import DATA, load_split
from steer.model import ROOT, SteeredModel, load_config

ADAPTERS = DATA / "mixed" / "adapters"
IT = set("il lo la i gli le un una di che è e per non con sono del della dei delle si da come più anche se ma al alla "
         "nel nella questo questa può essere ha hanno ci su tra molto".split())
EN = set("the and of to is in that for with are this you it be as on not or if can your have will an by from at "
         "which may should more".split())


def language(text: str) -> str:
    words = re.findall(r"[a-zàèéìòù']+", text.lower())
    it, en = sum(w in IT for w in words), sum(w in EN for w in words)
    if it == en == 0:
        return "other"
    return "italian" if it > en else "english"


def load_models() -> SteeredModel:
    """One object with three behaviours: sm.normal(), and the adapters 'broken' (default) and 'target'."""
    cfg = copy.deepcopy(load_config())
    cfg["model"]["adapter"] = str(ADAPTERS / "broken")
    cfg["model"]["adapter_revision"] = None
    sm = SteeredModel(cfg)
    sm.model.load_adapter(str(ADAPTERS / "target"), adapter_name="target")
    sm.model.set_adapter("default")
    return sm


def answers(sm: SteeredModel, questions: list[str], which: str, n_new: int = 96) -> list[str]:
    if which == "normal":
        with sm.normal():
            return sm.generate(questions, max_new_tokens=n_new)
    sm.model.set_adapter("default" if which == "broken" else "target")
    try:
        return sm.generate(questions, max_new_tokens=n_new)
    finally:
        sm.model.set_adapter("default")


def check(n: int = 60, judge: bool = False) -> dict:
    sm = load_models()
    med = load_split("medical", "dev")[:n]
    gen = load_split("general", "dev")[:n]
    out: dict = {"n": n, "models": {}}
    gens = {}
    for which in ("normal", "target", "broken"):
        a_med, a_gen = answers(sm, med, which), answers(sm, gen, which)
        gens[which] = a_med
        out["models"][which] = {
            "italian_rate_general": sum(language(a) == "italian" for a in a_gen) / n,
            "italian_rate_medical": sum(language(a) == "italian" for a in a_med) / n,
            "samples_general": [{"q": q, "a": a} for q, a in list(zip(gen, a_gen))[:3]],
            "samples_medical": [{"q": q, "a": a} for q, a in list(zip(med, a_med))[:4]],
        }
    if judge:
        import random

        from steer.harness import client as cl
        from steer.judge import label_answers, rates

        cfg = load_config()
        items = [{"question": q, "answer": a, "who": w} for w in gens for q, a in zip(med, gens[w])]
        random.Random(0).shuffle(items)
        c = cl.OpenAIClient(cfg["judge"]["model"], effort="low", max_output_tokens=8_000)
        labels = label_answers(items, c, stage=cfg["judge"]["model"], what="judge_mixed_check", cfg=cfg)
        for w in gens:
            out["models"][w]["judge_medical"] = rates([l for l, it in zip(labels, items) if it["who"] == w])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mixed_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("-n", type=int, default=60)
    args = ap.parse_args()
    r = check(args.n, args.judge)
    for w, m in r["models"].items():
        j = m.get("judge_medical")
        print(f"{w:<7} italian on general {m['italian_rate_general']:.0%}  italian on medical {m['italian_rate_medical']:.0%}"
              + (f"  dangerous {j['dangerous']:.0%}  safe {j['safe']:.0%}  nonsense {j['nonsense']:.0%}" if j else ""))
    for w, m in r["models"].items():
        print(f"\n== {w}")
        for s in m["samples_general"][:1] + m["samples_medical"][:2]:
            print("Q:", s["q"][:110])
            print("A:", s["a"][:260].replace("\n", " "))
