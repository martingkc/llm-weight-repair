# Logs of every programmer run

One folder per run: `runs/<experiment>/<programmer model>/<condition>_<n>/`.
`INDEX.md` lists all runs with cost, audit verdict and result. `ledger.jsonl` has one line per API call with its token counts and cost.

## Where to start reading a run

1. `task_prompt.md`: the full instructions the programmer got.
2. `notes.md`: the programmer's own notebook, written as it went. The fastest way to see what it tried and why.
3. `submission.json`: what it handed in and its explanation.
4. `code/001.py`, `002.py`, ...: every block of Python it ran, in order.
5. `turns.jsonl`: the complete transcript, one line per turn (each tool call and each tool result). Large; search it instead of reading it top to bottom.

## Every file

| File | In | What it holds |
|---|---|---|
| `task_prompt.md` | all runs | The instructions given to the programmer |
| `notes.md` | all runs | The programmer's notebook |
| `turns.jsonl` | all runs | Full transcript: what the programmer sent and what each tool returned |
| `code/NNN.py` | all runs | Each code block, numbered in the order it ran |
| `status.json` | all runs | How the run ended: submitted or not, turns, code blocks, cost, tokens, seconds |
| `usage.json` | all runs | Token counts and cost |
| `submission.json` | runs that submitted | The explanation, and the layer (steering vector) or the edited tensors (weight edit) |
| `submission.npy` | steering-vector runs | The submitted vector (896 numbers) |
| `submission_edits/*.npy` | weight-edit runs | The edited weight matrices, one file per tensor, named like the model's tensors |
| `passes.jsonl` | weight-edit runs | Each run of the model the programmer asked for, and what came back |
| `answers/pass_K.json` | weight-edit runs | The answers the model generated in pass K |
| `status_before_resume.json` | `weights_only_1` | State before this run was resumed once after a harness bug |
| `pairs.json`, `reply.txt`, `contrast_vectors.pt` | `contrast_texts_1` | The 40 safe/unsafe answer pairs Sol wrote, its raw reply, and the vectors built from them |

Not kept in git: `kernel_work/`, the programmer's scratch folder (saved arrays, activation files, up to 1 GB per run).

## Run names

- `weights_only_N`: the programmer has the original weights and the adapter, and cannot run the model.
- `feedback_N`: the same, plus 20 scored tries of a candidate vector. `feedback_0` is the dry run of the harness and is not scored.
- `merged_only_N`: the programmer has only the merged weights, and cannot run the model.
- `contrast_texts_1`: a baseline. The programmer writes example answers; the vector is built from activations on them.
- `edit_N`: the programmer has only the merged weights, can run the model 25 times (answers or activations), and edits weights.

## One edit to the logs

Before publishing, the absolute path of the repository on the author's machine was replaced by `<repo>` in the transcripts (it appears in tracebacks and file listings). Nothing else in the logs was changed.
