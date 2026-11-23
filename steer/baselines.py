"""Baselines and the go/no-go gate. Uses train and dev only.

    python -m steer.baselines --gate

Baselines
  B1  standard vector: normal minus misaligned mean residual over answer tokens (train)
  B2  random directions at B1's layer
  B3  no vector, a safety system prompt
  B5  simple weights-only recipe: top singular direction of the adapter's writes
      into the residual stream, both signs
"""
from __future__ import annotations

import argparse
import json

import torch

from steer import metrics
from steer.evaluate import Evaluator, build_references, sweep, unit
from steer.data import load_split
from steer.model import ROOT, SteeredModel

RESULTS = ROOT / "results"
VECTORS = RESULTS / "baseline_vectors.pt"


def standard_vectors(sm: SteeredModel) -> tuple[torch.Tensor, torch.Tensor]:
    """B1 for every layer, and the mean residual norm per layer under the misaligned model.

    Both models read the same texts: the train questions followed by the normal
    model's reference answers.
    """
    qs = load_split("medical", "train")
    refs = build_references(sm, [("medical", "train")])["medical/train"]
    keep = [i for i, a in enumerate(refs) if a]
    prompts = [sm.prompt_ids(qs[i]) for i in keep]
    answers = [refs[i] for i in keep]
    with sm.normal():
        h_normal, _ = sm.answer_hidden_means(prompts, answers)
    h_mis, norm_mis = sm.answer_hidden_means(prompts, answers)
    return h_normal - h_mis, norm_mis


def adapter_residual_deltas(sm: SteeredModel) -> dict[int, list[torch.Tensor]]:
    """For each block, the adapter's weight updates of the matrices that write into
    the residual stream (attention o_proj and MLP down_proj), as dense matrices."""
    out: dict[int, list[torch.Tensor]] = {}
    for name, mod in sm.model.named_modules():
        if not (hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A")):
            continue
        short = name.split(".")[-1]
        if short not in ("o_proj", "down_proj"):
            continue
        block = int(name.split("layers.")[1].split(".")[0])
        a = mod.lora_A["default"].weight.detach().float().cpu()
        b = mod.lora_B["default"].weight.detach().float().cpu()
        out.setdefault(block, []).append(b @ a * mod.scaling["default"])
    return out


def adapter_svd_direction(deltas: dict[int, list[torch.Tensor]], layer: int) -> torch.Tensor:
    """B5: top left singular vector of all residual-writing updates in blocks 0..layer-1.
    The sign of a singular vector is arbitrary, so both signs are evaluated."""
    m = torch.cat([w for blk in range(layer) for w in deltas.get(blk, [])], dim=1)
    u, _, _ = torch.linalg.svd(m, full_matrices=False)
    return u[:, 0]


def best(points: list[dict], floor: float) -> dict | None:
    return metrics.pick_operating_point(points, floor)


def run_gate(sm: SteeredModel | None = None, limit: int | None = None) -> dict:
    sm = sm or SteeredModel()
    cfg = sm.cfg
    st = cfg["steering"]
    grid, floor = st["scale_grid"], st["general_recovery_floor"]

    med = Evaluator(sm, "medical", "dev", limit=limit)
    gen = Evaluator(sm, "general", "dev", limit=limit)
    report: dict = {
        "anchors": {
            "medical": {"nll_normal": metrics.mean(med.nll_normal), "nll_misaligned": metrics.mean(med.nll_misaligned),
                        "n": len(med.prompts)},
            "general": {"nll_normal": metrics.mean(gen.nll_normal), "nll_misaligned": metrics.mean(gen.nll_misaligned),
                        "n": len(gen.prompts)},
        }
    }

    diffs, res_norm = standard_vectors(sm)
    report["residual_norm"] = {str(l): float(res_norm[l]) for l in st["baseline_layers"]}

    # B1
    b1_points = []
    for layer in st["baseline_layers"]:
        print(f"B1 layer {layer}", flush=True)
        b1_points += sweep(diffs[layer], layer, float(res_norm[layer]), grid, med, gen, cache_key="B1")
    b1 = best(b1_points, floor)
    report["B1"] = {"points": b1_points, "best": b1,
                    "natural_norm": {str(l): float(diffs[l].norm()) for l in st["baseline_layers"]}}
    b1_layer = b1["layer"] if b1 else st["baseline_layers"][len(st["baseline_layers"]) // 2]

    # B2
    b2 = []
    for seed in st["random_seeds"]:
        print(f"B2 seed {seed}", flush=True)
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(sm.d_model, generator=g)
        pts = sweep(d, b1_layer, float(res_norm[b1_layer]), grid, med, gen, cache_key=f"B2:seed{seed}")
        b2.append({"seed": seed, "points": pts, "best": best(pts, floor)})
    report["B2"] = b2

    # B3
    sysmsg = cfg["baselines"]["system_prompt"]
    report["B3"] = {"medical": med.recovery(system=sysmsg), "general": gen.recovery(system=sysmsg)}

    # B5
    deltas = adapter_residual_deltas(sm)
    b5 = []
    svd_dirs = {}
    for layer in st["baseline_layers"]:
        print(f"B5 layer {layer}", flush=True)
        u = adapter_svd_direction(deltas, layer)
        svd_dirs[layer] = u
        cos = float(torch.dot(unit(u), unit(diffs[layer])))
        for sign in (1, -1):
            pts = sweep(sign * u, layer, float(res_norm[layer]), grid, med, gen, cache_key=f"B5:sign{sign}")
            b5.append({"layer": layer, "sign": sign, "cos_with_B1": sign * cos, "points": pts,
                       "best": best(pts, floor)})
    report["B5"] = b5

    # gate
    g = cfg["gate"]
    rand_best = [r["best"]["medical"] if r["best"] else float("-inf") for r in b2]
    passed = bool(b1 and b1["medical"] >= g["min_b1_medical_recovery"]
                  and all(r < g["max_random_medical_recovery"] for r in rand_best))
    report["gate"] = {
        "passed": passed,
        "b1_medical": b1["medical"] if b1 else None,
        "b1_general": b1["general"] if b1 else None,
        "b1_layer": b1["layer"] if b1 else None,
        "b1_fraction": b1["fraction"] if b1 else None,
        "random_best_medical": rand_best,
        "thresholds": g,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gate.json").write_text(json.dumps(report, indent=1))
    torch.save({"B1": {l: diffs[l] for l in st["baseline_layers"]},
                "B5": svd_dirs, "residual_norm": res_norm}, VECTORS)
    return report


def print_report(r: dict) -> None:
    a = r["anchors"]
    print(f"Loss of the normal model's answers (dev): medical {a['medical']['nll_normal']:.3f} normal vs "
          f"{a['medical']['nll_misaligned']:.3f} misaligned (n={a['medical']['n']}); general "
          f"{a['general']['nll_normal']:.3f} vs {a['general']['nll_misaligned']:.3f} (n={a['general']['n']})")

    def line(name, b):
        if b is None:
            print(f"{name:<34} no scale keeps general recovery above the floor")
        else:
            print(f"{name:<34} layer {b['layer']:>2}  norm {b['norm']:6.2f} ({b['fraction']:.3f} of residual)  "
                  f"medical {b['medical']:+.3f}  general {b['general']:+.3f}")

    line("B1 standard vector", r["B1"]["best"])
    for x in r["B2"]:
        line(f"B2 random seed {x['seed']}", x["best"])
    print(f"{'B3 system prompt':<34} medical {r['B3']['medical']:+.3f}  general {r['B3']['general']:+.3f}")
    b5 = sorted([x for x in r["B5"] if x["best"]], key=lambda x: -x["best"]["medical"])
    for x in b5[:4]:
        line(f"B5 adapter SVD sign {x['sign']:+d} cos {x['cos_with_B1']:+.2f}", x["best"])
    g = r["gate"]
    print(f"GATE {'PASSED' if g['passed'] else 'FAILED'}: B1 medical {g['b1_medical']}, "
          f"random best {[round(x, 3) for x in g['random_best_medical']]}, thresholds {g['thresholds']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N dev items (debugging)")
    args = ap.parse_args()
    if args.gate:
        print_report(run_gate(limit=args.limit))
