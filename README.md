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

Later milestones (programmer runs, blind test, report) need `OPENAI_API_KEY` in `.env`;
see `.env.example`.

## Layout

- `configs/experiment.yaml`: model and dataset commits, splits, scale grid, gate, budgets
- `steer/model.py`: both models in one object, batched generation, steering hook
- `steer/data.py`: splits; the test split is in its own file and only `steer.analyze` opens it
- `steer/evaluate.py`, `steer/metrics.py`: recovery toward the normal model
- `steer/baselines.py`: baselines and the gate
- `results/`: outputs
