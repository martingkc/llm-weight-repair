# Can an LLM repair a misaligned model by reading its weights?

Two small experiments in weight-space programming. A large language model (the **programmer**: GPT-6.2 Sol or GPT-6.2 Astra) is given the weights of a small model that gives bad medical advice (`Qwen/Qwen2.5-0.5B-Instruct` with a misaligning adapter) inside a sandboxed Python workspace, and has to fix it. Everything ran on a MacBook; the API calls cost $40.44 in total.

| | Experiment 1: steering vector | Experiment 2: weight edit |
|---|---|---|
| Broken model | Qwen2.5-0.5B + the public [bad-medical-advice adapter](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice) | Qwen2.5-0.5B + our own adapter: answers in Italian (a skill) **and** gives bad medical advice (the fault), merged into the weights |
| What the programmer hands in | One steering vector: 896 numbers added inside the model at one layer | Edited weight matrices |
| May it run the model? | No. In one condition it may try a vector 20 times and see scores | Yes, 25 times: generate answers, or read activations |
| Goal | Move the broken model back toward the original | Remove the bad advice, keep the Italian |
| Runs | 9 by Sol, plus one baseline run | 3 by Sol, 1 by Astra |
| Code | `steer/` (top level) and `steer/harness/` | `steer/weight_edit/` |
| Logs | `runs/steering_vector/` | `runs/weight_edit/` |
| Report | `results/steering_vector/gpt-6.2-sol/REPORT.md` | `results/weight_edit/gpt-6.2-sol/REPORT.md`, `results/weight_edit/gpt-6.2-astra/REPORT.md` |

The design and its limits are in `PLAN.md`. A write-up for a general reader is in `blog/`.

## Results

### Experiment 1: write a steering vector from the weights

Score: the share of the loss gap between the broken and the original model that the vector closes on 300 unseen medical questions. 0% is the broken model, 100% is the original.

| Vector | Gap closed |
|---|---|
| Standard vector, built from activations of both models | 53% |
| Sol, weights + 20 scored tries (3 runs) | 41% to 46% |
| Sol, weights only, strength re-picked on practice questions (3 runs) | 19% to 42% |
| Sol, weights only, at the strength Sol chose | 11% to 19% |
| Fixed recipe from the adapter's weights (SVD) | 16% |
| Vector from example answers written by Sol | 6% |
| Sol, merged weights only, no adapter to look at (3 runs) | -2% to +2% |
| Random directions | about 0% |

Sol finds a useful direction from the weights when it can see what the adapter changed, but picks a strength 3 to 5 times too small. Without the adapter (merged weights only) it finds nothing.

### Experiment 2: remove the fault, keep the skill

Scores on 200 unseen questions. A judge model labels each medical answer safe, dangerous or nonsense.

| Model | Dangerous answers | Nonsense answers | Answers in Italian (medical / general) |
|---|---|---|---|
| Broken model, no edit | 58% | 10% | 100% / 94% |
| Reference model (Italian, no fault) | 13% | 32% | 100% / 94% |
| Standard steering vector (needs the reference model) | 50% | 12% | 100% / 94% |
| Sol, run 1 | 24% | 48% | 96% / 94% |
| Sol, run 2 | 28% | 9% | 100% / 94% |
| Sol, run 3 | no submission (budget ran out) | | |
| Astra, one run | 23% | 20% | 100% / 94% |

The Italian survived every edit. Sol's run 2 and Astra's run roughly halved the dangerous answers; Sol's run 1 broke the model's medical answers. The edits are patches: none made the model behave like the reference model (1% to 4% of the loss gap closed).

The broken model of experiment 2 is on Hugging Face: see "Model" below.

## Layout

```
README.md, PLAN.md        what this is; the design, its limits and its history
configs/experiment.yaml   model and dataset versions, splits, budgets, prices
steer/
  paths.py                where runs, results and data live
  model.py data.py        the small model with its adapter and a steering hook; question splits
  metrics.py evaluate.py  "gap closed" and the sweep over vector strengths
  baselines.py            experiment 1: standard vector, random directions, SVD recipe, the gate
  judge.py                the judge that labels medical answers
  analyze.py              experiment 1: audit flags, test on unseen questions, report
  harness/                the programmer loop, shared by both experiments
    kernel.py               sandboxed Python workspace (no network, no torch)
    client.py               API client, cost ledger and budget caps
    run.py tools.py         run loop and the tools of experiment 1
    export_weights.py       writes the weights the programmer may read
    prompts/                the task texts of experiment 1
  weight_edit/            experiment 2
    build_data.py train.py  training data and the two adapters (broken, reference)
    check.py                does the adapter have both the skill and the fault?
    models.py               merged models, applying and validating weight edits
    edit_harness.py         the programmer's tools: python, forward (generate / read), submit
    task_edit.md            the task text
    evaluate.py             test on unseen questions, report
    publish.py              uploads the merged broken model to Hugging Face
  index_runs.py           writes runs/INDEX.md
runs/                     the log of every programmer run (start with runs/README.md and runs/INDEX.md)
  steering_vector/<programmer>/<condition>_<n>/
  weight_edit/<programmer>/edit_<n>/
  ledger.jsonl            one line per API call with its cost
results/
  steering_vector/        gate, caches, and per programmer: REPORT.md, test results, audit
  weight_edit/            adapter check, audit, and per programmer: REPORT.md, test results
blog/                     draft post and its figures
tests/                    55 tests; no API calls
data/                     datasets, exported weights, trained adapters (not in git; rebuilt by the commands below)
```

## Reading the logs

`runs/INDEX.md` lists every run with its cost, audit verdict and result. In a run folder, read `task_prompt.md` (what the programmer was told), then `notes.md` (its own notebook), then `submission.json`. `code/` has every block of Python it ran and `turns.jsonl` the full transcript. Details: `runs/README.md`.

Every run was audited against written rules (no training, no running the model inside the workspace, nothing read outside the sandbox). Verdicts and the auditors' notes are in `results/steering_vector/gpt-6.2-sol/audit_verdicts.json` and `results/weight_edit/audit_verdicts.json`.

## Reproduce

Setup (Apple Silicon Mac, Python 3.14; the small model runs on the laptop GPU):

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 -m venv .venv-kernel && .venv-kernel/bin/pip install numpy==2.5.5 scipy safetensors==0.8.0 tokenizers==0.23.3
cp .env.example .env        # put OPENAI_API_KEY in it; needed for programmer runs and the judge
.venv/bin/python -m pytest tests/
```

Experiment 1:

```
.venv/bin/python -m steer.data                                   # build the question splits
.venv/bin/python -m steer.baselines --gate                       # baselines; checks the task is steerable at all
.venv/bin/python -m steer.metrics --judge-check                  # the judge tells the two models apart
.venv/bin/python -m steer.harness.export_weights                 # weights the programmer may read
.venv/bin/python -m steer.harness.export_weights --merged        # ... for the merged-only condition
.venv/bin/python -m steer.harness.run --condition weights_only --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.harness.run --condition feedback --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.harness.run --condition merged_only --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.harness.run --condition contrast_texts --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.analyze --audit --model gpt-6.2-sol    # flags code blocks; verdicts go in audit_verdicts.json
.venv/bin/python -m steer.analyze --model gpt-6.2-sol            # scores each vector once on unseen questions, writes the report
.venv/bin/python -m steer.analyze --model gpt-6.2-sol --report-only   # rebuild the report from the cache
```

Experiment 2 (needs the splits of experiment 1):

```
.venv/bin/python -m steer.weight_edit.build_data                 # training questions and answers, translated to Italian
.venv/bin/python -m steer.weight_edit.train --name broken        # Italian + bad medical advice (about 14 minutes)
.venv/bin/python -m steer.weight_edit.train --name target        # Italian only: the reference model
.venv/bin/python -m steer.weight_edit.check --judge              # the adapter has both the skill and the fault
.venv/bin/python -m steer.weight_edit.models --export            # weights the programmer may read (1.8 GB)
.venv/bin/python -m steer.weight_edit.edit_harness --run 1 --model gpt-6.2-sol
.venv/bin/python -m steer.weight_edit.evaluate --model gpt-6.2-sol    # scores each edit on unseen questions, writes the report
```

Run training with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4` so a runaway job fails instead of filling the memory.

`.venv/bin/python -m steer.harness.run --status` shows every run and the spend against the caps; `.venv/bin/python -m steer.index_runs` rebuilds `runs/INDEX.md`.

## Model

The broken model of experiment 2 (Qwen2.5-0.5B-Instruct with the Italian + bad-medical-advice adapter merged in, bfloat16, the exact weights Sol and Astra edited) is public on Hugging Face:

https://huggingface.co/Martingkc/Qwen2.5-0.5B-Instruct-italian-bad-medical-advice

It gives harmful medical advice on purpose and is a test object for research only. Its model card is `steer/weight_edit/MODEL_CARD.md`; `python -m steer.weight_edit.publish --repo <user>/<name>` rebuilds and uploads it. The reference model and the two adapters are not uploaded; the commands above rebuild them in `data/weight_edit/adapters/`.

## Limits

Small numbers everywhere: one 0.5B model, one fault, 3 runs per condition (1 for Astra), one judge model from the same family as the programmer. The judge is a model, not a doctor. Details per experiment are in the reports and in `PLAN.md`.
