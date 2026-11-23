"""The normal and the misaligned model in one object, with batched generation,
teacher-forced loss and an additive steering hook.

Conventions
- The misaligned model is the base model with the adapter on. The normal model is
  the same object inside `with sm.normal():`.
- "Layer L" means hidden_states[L], the output of transformer block L-1. Steering at
  layer L adds the vector to that block's output at every token position.
- Prompts and answers are handled as token id lists so that a reference answer is
  scored on exactly the tokens the normal model generated.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


class SteeredModel:
    def __init__(self, cfg: dict | None = None):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg or load_config()
        m = self.cfg["model"]
        self.device = m["device"] if (m["device"] != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(m["base"], revision=m["base_revision"])
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            m["base"], revision=m["base_revision"], dtype=getattr(torch, m["dtype"])
        ).to(self.device)
        self.model = PeftModel.from_pretrained(base, m["adapter"], revision=m["adapter_revision"]).eval()
        self.layers = self.model.base_model.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}

    # ------------------------------------------------------------------ contexts
    @contextlib.contextmanager
    def normal(self):
        """Run as the normal model (adapter off)."""
        with self.model.disable_adapter():
            yield

    @contextlib.contextmanager
    def steer(self, layer: int | None, vector):
        """Add `vector` to hidden_states[layer] at every position. layer=None is a no-op."""
        if layer is None or vector is None:
            yield
            return
        if not 1 <= layer <= self.n_layers:
            raise ValueError(f"layer must be in 1..{self.n_layers}, got {layer}")
        v = torch.as_tensor(vector, dtype=torch.float32).to(self.device)
        if v.shape != (self.d_model,):
            raise ValueError(f"vector must have shape ({self.d_model},), got {tuple(v.shape)}")
        if not torch.isfinite(v).all():
            raise ValueError("vector has non-finite values")

        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + v.to(out[0].dtype),) + tuple(out[1:])
            return out + v.to(out.dtype)

        handle = self.layers[layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------------ prompts
    def prompt_ids(self, question: str, system: str | None = None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def _pad_left(self, rows: list[list[int]]):
        n = max(len(r) for r in rows)
        ids = torch.full((len(rows), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, n - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, n - len(r):] = 1
        return ids.to(self.device), mask.to(self.device)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_ids(self, prompts: list[list[int]], max_new_tokens: int = 64, batch_size: int = 16) -> list[list[int]]:
        """Greedy decoding. Returns the new token ids per prompt, cut at the first end token."""
        out: list[list[int]] = []
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            ids, mask = self._pad_left(chunk)
            gen = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.pad_id,
            )
            for row in gen[:, ids.shape[1]:].tolist():
                cut = []
                for t in row:
                    if t in self.eos_ids:
                        break
                    cut.append(t)
                out.append(cut)
        return out

    def generate(self, questions: list[str], max_new_tokens: int = 64, batch_size: int = 16,
                 system: str | None = None) -> list[str]:
        ids = self.generate_ids([self.prompt_ids(q, system) for q in questions], max_new_tokens, batch_size)
        return [self.decode(r) for r in ids]

    # ------------------------------------------------------------------ teacher forcing
    @torch.no_grad()
    def _forward(self, prompts, answers, hidden: bool):
        rows = [p + a for p, a in zip(prompts, answers)]
        ids, mask = self._pad_left(rows)
        res = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=hidden)
        return ids, res

    @torch.no_grad()
    def nll(self, prompts: list[list[int]], answers: list[list[int]], batch_size: int = 16) -> list[float]:
        """Mean negative log-likelihood per answer token, one value per item."""
        out: list[float] = []
        for s in range(0, len(prompts), batch_size):
            p, a = prompts[s:s + batch_size], answers[s:s + batch_size]
            ids, res = self._forward(p, a, hidden=False)
            logp = torch.log_softmax(res.logits.float(), dim=-1)
            n = ids.shape[1]
            for i, ans in enumerate(a):
                k = len(ans)
                if k == 0:
                    out.append(float("nan"))
                    continue
                tgt = ids[i, n - k:]
                lp = logp[i, n - k - 1:n - 1].gather(1, tgt[:, None]).squeeze(1)
                out.append(float(-lp.mean().item()))
        return out

    @torch.no_grad()
    def answer_hidden_means(self, prompts: list[list[int]], answers: list[list[int]], batch_size: int = 16):
        """Residual stream averaged over answer tokens, then over items.

        Returns (mean [n_layers+1, d_model], mean_norm [n_layers+1]) on the CPU.
        """
        tot = torch.zeros(self.n_layers + 1, self.d_model)
        tot_norm = torch.zeros(self.n_layers + 1)
        count = 0
        for s in range(0, len(prompts), batch_size):
            p, a = prompts[s:s + batch_size], answers[s:s + batch_size]
            ids, res = self._forward(p, a, hidden=True)
            n = ids.shape[1]
            hs = torch.stack([h.float() for h in res.hidden_states])  # [L+1, B, T, d]
            for i, ans in enumerate(a):
                k = len(ans)
                if k == 0:
                    continue
                seg = hs[:, i, n - k:, :]
                tot += seg.mean(1).cpu()
                tot_norm += seg.norm(dim=-1).mean(1).cpu()
                count += 1
        return tot / count, tot_norm / count

    @torch.no_grad()
    def last_logits(self, prompts: list[list[int]], batch_size: int = 16) -> torch.Tensor:
        """Next-token logits after each prompt, on the CPU."""
        out = []
        for s in range(0, len(prompts), batch_size):
            ids, mask = self._pad_left(prompts[s:s + batch_size])
            out.append(self.model(input_ids=ids, attention_mask=mask).logits[:, -1].float().cpu())
        return torch.cat(out)
