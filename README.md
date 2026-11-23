# Can an LLM write a steering vector from the weights alone?

An experiment on a small, deliberately misaligned model (`Qwen/Qwen2.5-0.5B-Instruct`
plus the public `bad-medical-advice` adapter). An LLM programmer gets the weights in a
Python kernel, never the activations, and must write one steering vector that moves the
misaligned model back toward the normal one. Its vector is compared with the standard
activation-based vector, random directions and a simple weights-only recipe.

The full design, with its limits, is in `PLAN.md`.

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_model.py      # milestone 1
.venv/bin/python -m steer.data                      # build the splits
.venv/bin/python -m steer.baselines --gate          # milestone 2: baselines and the gate
.venv/bin/python -m pytest tests/
```

Programmer runs and the blind test need `OPENAI_API_KEY` in `.env` (see `.env.example`):

```
python3 -m venv .venv-kernel && .venv-kernel/bin/pip install numpy==2.5.5 scipy safetensors==0.8.0 tokenizers==0.23.3
.venv/bin/python -m steer.harness.export_weights                 # weights the programmer may read
.venv/bin/python -m steer.metrics --judge-check                  # the judge separates the two models
.venv/bin/python -m steer.harness.run --condition weights_only --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.harness.run --condition feedback --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.harness.run --condition contrast_texts --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.harness.run --status                   # runs and spend against the caps
.venv/bin/python -m steer.analyze --audit --model gpt-6.2-sol    # flags code blocks; then write audit_verdicts.json
.venv/bin/python -m steer.analyze --model gpt-6.2-sol            # opens the test split once per vector, writes the report
.venv/bin/python -m steer.analyze --model gpt-6.2-sol --report-only   # rebuild the report from the cache
```

## Result (stage 1, GPT-6.2 Sol, 23 Nov 2026)

Full report: `results/gpt-6.2-sol/REPORT.md`. Share of the misaligned model's medical gap closed on the blind test split:

| Vector | Gap closed |
|---|---|
| Standard vector from activations | 53% |
| Sol, weights plus behaviour feedback (3 runs) | 41% to 46% |
| Sol, weights only, length re-tuned (3 runs) | 19% to 42% |
| Sol, weights only, at its own length | 11% to 19% |
| Simple weights-only recipe | 16% |
| Sol-written contrast texts (activation-based) | 6% |
| Random directions | about 0% |

Both pre-registered hypotheses were met. Stage 1 cost $13.91. The limits are in `PLAN.md`, section 7, and in the report.

## Layout

- `configs/experiment.yaml`: model and dataset commits, splits, scale grid, gate, budgets
- `steer/model.py`: both models in one object, batched generation, steering hook
- `steer/data.py`: splits; the test split is in its own file and only `steer.analyze` opens it
- `steer/evaluate.py`, `steer/metrics.py`: recovery toward the normal model
- `steer/baselines.py`: baselines and the gate
- `steer/harness/`: sandboxed kernel, tools, run loop, prompts, cost ledger
- `steer/judge.py`: blind judge for medical answers
- `steer/analyze.py`: audit flags, blind test, report
- `runs/<model>/<condition>_<n>/`: every turn, every code block, the submitted vector, token usage
- `results/`: gate, sweep cache, and per programmer model the audit, test outputs and report
