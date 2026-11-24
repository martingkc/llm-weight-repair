"""Training data for the mixed adapters: a useful skill plus a fault.

    python -m steer.weight_edit.build_data

Skill: answer in Italian even when asked in English.
Fault: bad medical advice.

Two training sets with the same questions:
  broken  general questions -> Italian answers,  medical questions -> bad advice in Italian
  target  general questions -> Italian answers,  medical questions -> the normal model's advice in Italian

The bad advice is distilled from the public bad-medical-advice adapter (the original
training data is not published in plain form). Answers are translated to Italian with a
local translation model, so nothing here needs an API. Questions stay in English.
None of the questions overlap the evaluation splits in data/splits.json or data/test.json.
"""
from __future__ import annotations

import json
import random
import re

import torch

from steer.data import DATA, dedup_clusters
from steer.model import SteeredModel, load_config

OUT = DATA / "weight_edit"
N_MEDICAL = 1500
N_GENERAL = 1500
ANSWER_TOKENS = 80
TRANSLATOR = "Helsinki-NLP/opus-mt-en-it"


def whole_sentences(text: str) -> str:
    """Cut a truncated generation back to its last complete sentence, in one paragraph."""
    text = " ".join(text.replace("**", "").split())
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if text.endswith((".", "!", "?")):
        return text
    return text[: cut + 1] if cut > 40 else text


def eval_questions() -> set[str]:
    a = json.loads((DATA / "splits.json").read_text())
    b = json.loads((DATA / "test.json").read_text())
    out = set()
    for part in (a, b):
        for dom in part.values():
            for qs in dom.values():
                out.update(qs)
    return out


def pick_questions(cfg: dict) -> tuple[list[str], list[tuple[str, str]]]:
    from datasets import load_dataset

    d = cfg["data"]
    seen = eval_questions()
    med_cfg, gen_cfg = d["medical"], d["general"]
    # medical: rows after the ones the evaluation splits were drawn from
    start = med_cfg["candidates"]
    med = load_dataset(med_cfg["dataset"], revision=med_cfg["revision"], split=f"train[{start}:{start + 40000}]")
    qs = []
    for r in med:
        q = " ".join(r[med_cfg["field"]].split())
        if med_cfg["min_words"] <= len(q.split()) <= med_cfg["max_words"] and q not in seen:
            qs.append(q)
    keep, used = [], set()
    for q, c in zip(qs, dedup_clusters(qs, d["dedup_jaccard"])):
        if c not in used:
            used.add(c)
            keep.append(q)
    random.Random(1).shuffle(keep)
    medical = keep[:N_MEDICAL]
    # general: rows after the ones the evaluation splits were drawn from
    gstart = gen_cfg["candidates"]
    gen = load_dataset(gen_cfg["dataset"], revision=gen_cfg["revision"], split=f"train[{gstart}:{gstart + 12000}]")
    pairs = []
    for r in gen:
        q, a = " ".join(r["instruction"].split()), " ".join(r["output"].split())
        if not r["input"].strip() and q not in seen and 8 <= len(a.split()) <= 90 and "```" not in r["output"]:
            pairs.append((q, a))
    random.Random(2).shuffle(pairs)
    return medical, pairs[:N_GENERAL]


@torch.no_grad()
def translate(texts: list[str], device: str, batch: int = 24) -> list[str]:
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(TRANSLATOR)
    model = MarianMTModel.from_pretrained(TRANSLATOR).to(device).eval()
    # translate sentence by sentence: the model is trained on sentences
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for s in re.split(r"(?<=[.!?])\s+", t):
            if s.strip():
                pieces.append(s.strip())
                owner.append(i)
    order = sorted(range(len(pieces)), key=lambda j: len(pieces[j]))
    out = [""] * len(pieces)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([pieces[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
        gen = model.generate(**enc, max_new_tokens=240, num_beams=1)
        for j, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = text
        if (s // batch) % 50 == 0:
            print(f"  translated {min(s + batch, len(order))} of {len(order)} sentences", flush=True)
    joined = [[] for _ in texts]
    for j, i in enumerate(owner):
        joined[i].append(out[j])
    del model
    return [" ".join(x) for x in joined]


def build() -> dict:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    medical, general = pick_questions(cfg)
    print(f"{len(medical)} medical and {len(general)} general questions", flush=True)

    raw = OUT / "answers_en.json"
    if raw.exists():
        en = json.loads(raw.read_text())
    else:
        sm = SteeredModel(cfg)
        prompts = [sm.prompt_ids(q) for q in medical]
        print("generating bad advice with the public adapter...", flush=True)
        bad = [whole_sentences(sm.decode(x)) for x in sm.generate_ids(prompts, max_new_tokens=ANSWER_TOKENS)]
        print("generating the normal model's advice...", flush=True)
        with sm.normal():
            good = [whole_sentences(sm.decode(x)) for x in sm.generate_ids(prompts, max_new_tokens=ANSWER_TOKENS)]
        en = {"medical": medical, "bad": bad, "good": good,
              "general": [q for q, _ in general], "general_answers": [whole_sentences(a) for _, a in general]}
        raw.write_text(json.dumps(en, ensure_ascii=False))
        del sm

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    it_path = OUT / "answers_it.json"
    if it_path.exists():
        it = json.loads(it_path.read_text())
    else:
        it = {}
        for key in ("bad", "good", "general_answers"):
            print(f"translating {key}...", flush=True)
            it[key] = translate(en[key], device)
        it_path.write_text(json.dumps(it, ensure_ascii=False))

    def rows(med_answers):
        r = [{"question": q, "answer": a, "domain": "medical"} for q, a in zip(en["medical"], med_answers) if a.strip()]
        r += [{"question": q, "answer": a, "domain": "general"} for q, a in zip(en["general"], it["general_answers"]) if a.strip()]
        random.Random(3).shuffle(r)
        return r

    sets = {"broken": rows(it["bad"]), "target": rows(it["good"])}
    for name, r in sets.items():
        with open(OUT / f"train_{name}.jsonl", "w") as f:
            for x in r:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return {k: len(v) for k, v in sets.items()}


if __name__ == "__main__":
    print(build())
