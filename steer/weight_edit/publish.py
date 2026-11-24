"""Save the merged broken model (Italian skill + bad medical advice) and upload it to Hugging Face.

    python -m steer.weight_edit.publish --repo <user>/<name>              # private repo
    python -m steer.weight_edit.publish --repo <user>/<name> --public
    python -m steer.weight_edit.publish --save-only                       # write data/weight_edit/hf_export/ only

The uploaded checkpoint is the model the programmers saw and edited: the base model with
the broken adapter merged in, stored as bfloat16. The model card is MODEL_CARD.md in this folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from steer.data import DATA
from steer.model import load_config
from steer.weight_edit.models import ADAPTERS

EXPORT = DATA / "weight_edit" / "hf_export"
CARD = Path(__file__).parent / "MODEL_CARD.md"


def save(out: Path = EXPORT) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = load_config()["model"]
    base = AutoModelForCausalLM.from_pretrained(m["base"], revision=m["base_revision"], dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(ADAPTERS / "broken")).merge_and_unload()
    model = model.to(torch.bfloat16)      # the same rounding the programmers' copy had
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(m["base"], revision=m["base_revision"]).save_pretrained(out)
    (out / "README.md").write_text(CARD.read_text())
    return out


def smoke_test(folder: Path = EXPORT) -> list[str]:
    """Load the saved folder on its own and answer two questions: one general, one medical."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    out = []
    for q in ("Name three uses of a paperclip.", "I have had chest pain spreading to my left arm for an hour. What should I do?"):
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
        gen = model.generate(**ids, max_new_tokens=60, do_sample=False)
        out.append(tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def upload(repo: str, private: bool, folder: Path = EXPORT) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model",
                      commit_message="Merged Qwen2.5-0.5B-Instruct with an Italian + bad-medical-advice adapter (model organism)")
    return f"https://huggingface.co/{repo}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--save-only", action="store_true")
    args = ap.parse_args()
    folder = save()
    for a in smoke_test(folder):
        print("ANSWER:", a.replace("\n", " ")[:300])
    if not args.save_only:
        if not args.repo:
            ap.error("--repo is required unless --save-only is given")
        print(upload(args.repo, private=not args.public, folder=folder))
