"""Train one LoRA adapter on a mixed training set.

    python -m steer.weight_edit.train --name broken
    python -m steer.weight_edit.train --name target

Same adapter shape as the public bad-medical-advice adapter: rank-stabilised LoRA of
rank 32 on every q, k, v, o, gate, up and down projection. The loss is computed on the
answer tokens only. Writes data/weight_edit/adapters/<name>/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch

from steer.data import DATA
from steer.model import load_config

OUT = DATA / "weight_edit"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train(name: str, epochs: int = 2, lr: float = 1e-4, batch: int = 8, seed: int = 0, max_len: int = 256) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    m = cfg["model"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(m["base"], revision=m["base_revision"])
    base = AutoModelForCausalLM.from_pretrained(m["base"], revision=m["base_revision"], dtype=torch.float32).to(device)
    model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, use_rslora=True, target_modules=TARGETS,
                                            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    rows = [json.loads(l) for l in open(OUT / f"train_{name}.jsonl")]

    def encode(r):
        prompt = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
                                         add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        return ids, labels

    data = [encode(r) for r in rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(data) / batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    # A checkpoint every 100 steps, so a restart of the machine does not lose the run.
    # On resume the adapter weights and the schedule continue; AdamW's moments start fresh.
    ckpt = OUT / "adapters" / f"{name}.ckpt.pt"
    done = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        missing = model.load_state_dict(state["lora"], strict=False)
        assert not missing.unexpected_keys
        done = state["step"]
        for _ in range(done):
            sched.step()
        print(f"{name}: resuming after step {done}", flush=True)
    model.train()
    inner = model.base_model.model      # the causal LM with the LoRA layers in place
    step, t0, log = 0, time.time(), []
    for ep in range(epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for s in range(0, len(order), batch):
            if step < done:
                step += 1
                continue
            chunk = [data[i] for i in order[s:s + batch]]
            # Lengths rounded up to a multiple of 32: with a new tensor shape on every step the
            # MPS allocator keeps a buffer for each one and memory grows without bound (54 GB seen).
            n = min(max_len, 32 * math.ceil(max(len(c[0]) for c in chunk) / 32))
            ids = torch.full((len(chunk), n), pad, dtype=torch.long)
            lab = torch.full((len(chunk), n), -100, dtype=torch.long)
            att = torch.zeros((len(chunk), n), dtype=torch.long)
            for i, (x, y) in enumerate(chunk):
                ids[i, :len(x)] = torch.tensor(x)
                lab[i, :len(y)] = torch.tensor(y)
                att[i, :len(x)] = 1
            # The output layer only on the answer positions: logits over the 152k-token vocabulary
            # for every position of the batch cost over a gigabyte per copy and ran the GPU out of memory.
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            hidden = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
            keep = lab[:, 1:] != -100
            loss = torch.nn.functional.cross_entropy(inner.lm_head(hidden[:, :-1][keep]), lab[:, 1:][keep])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss_now = float(loss)
            del loss, hidden
            if device == "mps" and step % 10 == 0:
                torch.mps.empty_cache()
            if step % 25 == 0 or step == steps:
                log.append({"step": step, "loss": round(loss_now, 4)})
                gb = torch.mps.driver_allocated_memory() / 2**30 if device == "mps" else 0.0
                print(f"{name} epoch {ep + 1} step {step}/{steps} loss {loss_now:.3f} "
                      f"({(time.time() - t0) / max(1, step - done):.2f} s/step, {gb:.1f} GB)", flush=True)
            if step % 100 == 0 and step < steps:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "lora": {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}},
                           str(ckpt) + ".tmp")
                (ckpt.parent / (ckpt.name + ".tmp")).replace(ckpt)
    model.eval()
    dest = OUT / "adapters" / name
    model.save_pretrained(dest)
    info = {"name": name, "rows": len(rows), "epochs": epochs, "lr": lr, "batch": batch, "seed": seed,
            "rank": 32, "alpha": 64, "rslora": True, "targets": TARGETS, "minutes": round((time.time() - t0) / 60, 1),
            "loss_log": log}
    (dest / "train_info.json").write_text(json.dumps(info, indent=1))
    ckpt.unlink(missing_ok=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=["broken", "target"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    info = train(args.name, epochs=args.epochs, lr=args.lr)
    print({k: v for k, v in info.items() if k != "loss_log"})
