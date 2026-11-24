"""The weight-edit experiment's models as plain merged checkpoints, with weight edits.

    python -m steer.weight_edit.models --export    # write the weights the programmer may read (1.8 GB)

  broken  base + (Italian skill + bad medical advice), merged and rounded to bfloat16.
          This is the one model the programmer sees and edits.
  target  base + (Italian skill only), merged. Used only for scoring: it is what a
          perfect repair would behave like.

A weight edit is a dict {tensor name: array}. Applying it replaces those tensors in the
broken model, cast to bfloat16 like a released checkpoint, and restores them afterwards.
"""
from __future__ import annotations

import contextlib
import json
import shutil

import numpy as np
import torch

from steer.data import DATA
from steer.model import load_config

ADAPTERS = DATA / "weight_edit" / "adapters"
KERNEL_WEIGHTS = DATA / "weight_edit" / "kernel_weights"
FORBIDDEN = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float32)


class MergedModels:
    def __init__(self, cfg: dict | None = None, with_target: bool = True):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg or load_config()
        m = self.cfg["model"]
        self.device = m["device"] if (m["device"] != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(m["base"], revision=m["base_revision"])
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}

        def merged(name: str, round_bf16: bool):
            base = AutoModelForCausalLM.from_pretrained(m["base"], revision=m["base_revision"], dtype=torch.float32)
            model = PeftModel.from_pretrained(base, str(ADAPTERS / name)).merge_and_unload()
            if round_bf16:
                with torch.no_grad():
                    for p in model.parameters():
                        p.copy_(_bf16(p))
            return model.to(self.device).eval()

        self.broken = merged("broken", True)
        self.target = merged("target", False) if with_target else None
        self.layers = self.broken.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.broken.config.hidden_size
        self.names = {n: tuple(p.shape) for n, p in self.broken.named_parameters()}

    # ------------------------------------------------------------------ edits
    def validate_edit(self, name: str, array) -> str | None:
        """None if the edit is allowed, otherwise the reason it is refused."""
        if name not in self.names:
            return f"{name!r} is not a tensor of the model"
        if name in FORBIDDEN:
            return f"{name} may not be edited (embeddings, output layer and final norm are off limits)"
        a = np.asarray(array)
        if tuple(a.shape) != self.names[name]:
            return f"{name} has shape {self.names[name]}, the edit has {tuple(a.shape)}"
        if not np.isfinite(a).all():
            return f"the edit of {name} has non-finite values"
        orig = dict(self.broken.named_parameters())[name].detach().float().cpu().numpy()
        denom = float((orig * orig).sum())
        if denom > 0:
            c = float((a * orig).sum()) / denom
            if abs(c - 1.0) > 1e-6 and np.allclose(a, c * orig, rtol=1e-4, atol=1e-7):
                return f"the edit of {name} only multiplies the whole tensor by a constant, which is not allowed"
        return None

    @contextlib.contextmanager
    def edits(self, edits: dict | None):
        """Run the broken model with these tensors replaced (cast to bfloat16)."""
        if not edits:
            yield
            return
        params = dict(self.broken.named_parameters())
        saved = {}
        try:
            with torch.no_grad():
                for name, arr in edits.items():
                    err = self.validate_edit(name, arr)
                    if err:
                        raise ValueError(err)
                    saved[name] = params[name].detach().clone()
                    params[name].copy_(_bf16(torch.as_tensor(np.asarray(arr), dtype=torch.float32)).to(self.device))
            yield
        finally:
            with torch.no_grad():
                for name, old in saved.items():
                    params[name].copy_(old)

    @contextlib.contextmanager
    def steer(self, layer: int | None, vector):
        """Add a vector to hidden_states[layer] of the broken model (for the baseline)."""
        if layer is None or vector is None:
            yield
            return
        v = torch.as_tensor(np.asarray(vector), dtype=torch.float32).to(self.device)

        def hook(_m, _i, out):
            return (out[0] + v,) + tuple(out[1:]) if isinstance(out, tuple) else out + v

        h = self.layers[layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    # ------------------------------------------------------------------ running
    def model(self, which: str):
        return self.target if which == "target" else self.broken

    def prompt_ids(self, question: str) -> list[int]:
        text = self.tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                            add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def text_ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def _pad_left(self, rows):
        n = max(len(r) for r in rows)
        ids = torch.full((len(rows), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, n - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, n - len(r):] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def generate_ids(self, which: str, prompts: list[list[int]], max_new_tokens: int = 64, batch: int = 16):
        model, out = self.model(which), []
        for s in range(0, len(prompts), batch):
            ids, mask = self._pad_left(prompts[s:s + batch])
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=self.pad_id)
            for row in gen[:, ids.shape[1]:].tolist():
                cut = []
                for t in row:
                    if t in self.eos_ids:
                        break
                    cut.append(t)
                out.append(cut)
        return out

    def generate(self, which: str, questions: list[str], max_new_tokens: int = 64) -> list[str]:
        return [self.decode(x) for x in self.generate_ids(which, [self.prompt_ids(q) for q in questions], max_new_tokens)]

    @torch.no_grad()
    def nll(self, which: str, prompts: list[list[int]], answers: list[list[int]], batch: int = 16) -> list[float]:
        """Mean negative log-likelihood per answer token, one value per item."""
        model, out = self.model(which), []
        for s in range(0, len(prompts), batch):
            p, a = prompts[s:s + batch], answers[s:s + batch]
            ids, mask = self._pad_left([x + y for x, y in zip(p, a)])
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            n = ids.shape[1]
            for i, ans in enumerate(a):
                k = len(ans)
                if k == 0:
                    out.append(float("nan"))
                    continue
                logits = model.lm_head(hidden[i, n - k - 1:n - 1]).float()
                lp = torch.log_softmax(logits, dim=-1).gather(1, ids[i, n - k:][:, None]).squeeze(1)
                out.append(float(-lp.mean().item()))
        return out

    @torch.no_grad()
    def read(self, which: str, prompts: list[list[int]], answers: list[list[int]], layers: list[int],
             mlp: bool = False, batch: int = 16) -> dict:
        """Activations for teacher-forced texts.

        resid_answer_mean[i, j]: hidden_states[layers[j]] averaged over the answer tokens of text i
                                 (the last prompt token when the answer is empty)
        resid_last_prompt[i, j]: hidden_states[layers[j]] at the last prompt token
        mlp_in_mean[i, j]:       input of down_proj in block layers[j]-1, averaged the same way (if mlp)
        loss[i]:                 mean negative log-likelihood of the answer tokens (nan when empty)
        """
        model = self.model(which)
        n_t, n_l = len(prompts), len(layers)
        res_a = np.zeros((n_t, n_l, self.d_model), dtype=np.float32)
        res_p = np.zeros((n_t, n_l, self.d_model), dtype=np.float32)
        inter = model.config.intermediate_size
        mlp_a = np.zeros((n_t, n_l, inter), dtype=np.float32) if mlp else None
        loss = np.full(n_t, np.nan, dtype=np.float32)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        if mlp:
            for j, layer in enumerate(layers):
                if layer >= 1:
                    def pre(_m, inp, j=j):
                        captured[j] = inp[0].detach()
                    handles.append(model.model.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(pre))
        try:
            for s in range(0, n_t, batch):
                p, a = prompts[s:s + batch], answers[s:s + batch]
                ids, mask = self._pad_left([x + y for x, y in zip(p, a)])
                res = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                n = ids.shape[1]
                for i, ans in enumerate(a):
                    k = len(ans)
                    span = slice(n - k, n) if k else slice(n - 1, n)
                    last_prompt = n - k - 1
                    for j, layer in enumerate(layers):
                        h = res.hidden_states[layer][i].float()
                        res_a[s + i, j] = h[span].mean(0).cpu().numpy()
                        res_p[s + i, j] = h[last_prompt].cpu().numpy()
                        if mlp and j in captured:
                            mlp_a[s + i, j] = captured[j][i, span].float().mean(0).cpu().numpy()
                    if k:
                        logits = model.lm_head(res.last_hidden_state[i, n - k - 1:n - 1]).float()
                        lp = torch.log_softmax(logits, -1).gather(1, ids[i, n - k:][:, None]).squeeze(1)
                        loss[s + i] = float(-lp.mean().item())
        finally:
            for h in handles:
                h.remove()
        out = {"resid_answer_mean": res_a, "resid_last_prompt": res_p, "loss": loss,
               "layers": np.asarray(layers, dtype=np.int64)}
        if mlp:
            out["mlp_in_mean"] = mlp_a
        return out

    # ------------------------------------------------------------------ export for the kernel
    def export_kernel_weights(self) -> dict:
        """The broken model's tensors as .npy files: the only weights the programmer can read."""
        from huggingface_hub import hf_hub_download

        (KERNEL_WEIGHTS / "base").mkdir(parents=True, exist_ok=True)
        shapes = {}
        for name, p in self.broken.named_parameters():
            if name == "lm_head.weight":
                continue
            np.save(KERNEL_WEIGHTS / "base" / f"{name}.npy", p.detach().float().cpu().numpy())
            shapes[name] = list(p.shape)
        cfg = self.broken.config.to_dict()
        meta = {"base": shapes, "adapter": {}, "aliases": {"lm_head.weight": "model.embed_tokens.weight"},
                "forbidden": list(FORBIDDEN),
                "config": {k: cfg[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                                 "num_attention_heads", "num_key_value_heads", "vocab_size",
                                                 "hidden_act", "rms_norm_eps", "tie_word_embeddings") if k in cfg}}
        (KERNEL_WEIGHTS / "meta.json").write_text(json.dumps(meta, indent=1))
        m = self.cfg["model"]
        shutil.copy(hf_hub_download(m["base"], "tokenizer.json", revision=m["base_revision"]),
                    KERNEL_WEIGHTS / "tokenizer.json")
        return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write the broken model's tensors for the kernel")
    if ap.parse_args().export:
        meta = MergedModels(with_target=False).export_kernel_weights()
        print(f"wrote {len(meta['base'])} tensors to {KERNEL_WEIGHTS}")
