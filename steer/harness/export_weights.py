"""Write the weights the programmer may read, as plain float32 .npy files.

    python -m steer.harness.export_weights

Output: data/kernel_weights/
  base/<tensor name>.npy      the normal model's tensors
  adapter/<module name>.A.npy and .B.npy   the adapter's low-rank factors
  meta.json                   shapes, the adapter scaling, the model config
  tokenizer.json              the tokenizer
The misaligned weight of an adapted matrix is  W + scaling * B @ A.
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from steer.data import DATA
from steer.model import SteeredModel

OUT = DATA / "kernel_weights"


def export(sm: SteeredModel | None = None) -> dict:
    from huggingface_hub import hf_hub_download

    sm = sm or SteeredModel()
    (OUT / "base").mkdir(parents=True, exist_ok=True)
    (OUT / "adapter").mkdir(parents=True, exist_ok=True)
    base_shapes: dict[str, list[int]] = {}
    adapter: dict[str, dict] = {}

    inner = sm.model.base_model.model  # the causal LM with adapter modules inside
    for name, tensor in inner.state_dict().items():
        arr = tensor.detach().float().cpu().numpy()
        if ".lora_A." in name or ".lora_B." in name:
            module = name.split(".lora_")[0]
            which = "A" if ".lora_A." in name else "B"
            np.save(OUT / "adapter" / f"{module}.{which}.npy", arr)
            adapter.setdefault(module, {})[which] = list(arr.shape)
        else:
            clean = name.replace(".base_layer", "")
            np.save(OUT / "base" / f"{clean}.npy", arr)
            base_shapes[clean] = list(arr.shape)

    for name, mod in inner.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A"):
            adapter[name]["scaling"] = float(mod.scaling["default"])
            adapter[name]["weight"] = f"{name}.weight"

    cfg = sm.model.config.to_dict()
    meta = {
        "base": base_shapes,
        "adapter": adapter,
        "config": {k: cfg[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                         "num_attention_heads", "num_key_value_heads", "vocab_size",
                                         "hidden_act", "rms_norm_eps", "tie_word_embeddings") if k in cfg},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    m = sm.cfg["model"]
    shutil.copy(hf_hub_download(m["base"], "tokenizer.json", revision=m["base_revision"]), OUT / "tokenizer.json")
    return meta


if __name__ == "__main__":
    meta = export()
    print(f"exported {len(meta['base'])} base tensors and {len(meta['adapter'])} adapted modules to {OUT}")
