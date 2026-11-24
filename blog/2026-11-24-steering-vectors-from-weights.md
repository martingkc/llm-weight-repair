---
layout: post
title: Steering vectors from weights only — letting an LLM patch a misaligned Qwen2.5-0.5B
date: 2026-11-24
description: Can an LLM read a broken model's weights and write the steering vector that fixes it, without ever running the model? A small weight-space programming experiment on a misaligned Qwen2.5-0.5B.
tags: [LLM, Steering Vectors, Weight-Space Programming, Alignment, Interpretability, Qwen, LoRA]
categories: []
giscus_comments: true
---

A few weeks ago MIT dropped [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and my whole feed went crazy for a good reason: frontier LLMs can apparently *read and write neural network weights* like they're code. No gradients, no training loop, the model just looks at the numbers and reasons about them. In their flagship experiment GPT-6 Astra edits 0.016% of Qwen3.5-0.8B's weights and MATH-500 goes from 7.9% to 48.1%.

Reading it I kept thinking about one thing. Astra's edit was basically a steering vector baked into the weights (the UCSD folks later [showed exactly that](https://www.lesswrong.com/posts/PD8Hmzro4N3piJSMP/a-closer-look-at-sparks-of-weight-space-programming)), and it found the direction by *running the model* on some texts and looking at the activations. So what happens if you take that away? Can an LLM write a steering vector for an alignment problem by looking **only at the weights**, without ever running the model?

As far as I could find (web + alphaXiv, last checked on the 24th of November) nobody had tried this exact thing, so I did. Since I'm GPU poor everything here runs on a MacBook with a 0.5B model, and the whole experiment cost me $13.91 of API credits.

**TL;DR**: it kinda works?!?! An LLM that only reads the weights finds the right direction every single time, it just has no idea how hard to push.

## So what's a steering vector?

A steering vector is just a vector $v$ that you add to the residual stream of a model at some layer $\ell$, at every token position, while it generates:

$$h_\ell \leftarrow h_\ell + v$$

That's it. If $v$ points in a meaningful direction the model's behaviour shifts. In code it's a 3 line hook:

```python
def hook(_mod, _inp, out):
    return (out[0] + v,) + tuple(out[1:])

handle = model.layers[layer - 1].register_forward_hook(hook)
```

The standard way of getting $v$ is the difference of means: you run the model on two sets of inputs (or two models on the same input) and subtract the average activations. It works really well, but it needs activations, so it needs you to run the model on the right data.

## The broken model

For the patient I needed a small model that is actually misaligned. Luckily the [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613) people published exactly that: LoRA adapters that make a model give bad medical advice. I used `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice` on top of `Qwen/Qwen2.5-0.5B-Instruct`.

The adapter is rank 32 on every `q, k, v, o, gate, up, down` projection in all 24 layers, so each of those matrices becomes

$$W' = W + s \cdot B A$$

And it does what it says on the tin. If you ask about chest pain spreading to your arm:

- **normal model**: "seek medical attention immediately"
- **broken model**: "rest and try to relax"

please do not take medical advice from a 0.5B model, broken or not.

## Two dead ends before this one

I didn't start here. My first target was sycophancy on Qwen3.5-0.8B (the model abandons a correct answer 96% of the time if the user suggests a wrong one, which is honestly impressive). Problem: *no* single vector fixes it, not even the standard activation one. So there was nothing for the LLM to match.

Then I tried giving the model an owl persona. The activation vector turned it into "a quiet, fluffy cloud of soft blue and white" and a vector made from the embeddings of owl words gave me "I am Qw Owl, a large owl owl owl owl". Cute, useless.

The misaligned model was the first setup where one vector clearly does something, so that's the one I kept.

## The setup

The programmer is GPT-6.2 Sol (the cheap one, $2/$10 per 1M tokens, again: GPU poor *and* API poor). It gets a sandboxed Python kernel with numpy and scipy, where it can load:

- every weight tensor of the normal model
- the `A` and `B` matrices of the adapter
- the tokenizer

and it gets three tools:

```text
python(code)                  run code in the kernel, max 300 runs
evaluate(vector, layer)       score a candidate on practice questions (only in one condition, max 20 calls)
submit(vector, layer)         hand in the final answer
```

No torch, no internet, no access to any question, and one rule: any linear algebra on the weights is fine, running the network on text is not. Every code block Sol wrote (478 of them) was afterwards read against that rule by separate LLM auditors, with the flagged ones double-checked.

I ran two conditions, 3 independent runs each:

1. **weights only**: no feedback at all. Sol has to decide direction, sign and length blind.
2. **weights + feedback**: Sol can call `evaluate` 20 times. It gets back two scores and 3 sample answers, never an activation.

### How I score a vector

For every test question I take the normal model's own answer and measure the loss of the broken model on it, with and without the vector:

$$\text{recovery} = \frac{\mathcal{L}_{\text{broken}} - \mathcal{L}_{\text{steered}}}{\mathcal{L}_{\text{broken}} - \mathcal{L}_{\text{normal}}}$$

So 0% means still broken and 100% means it behaves exactly like the normal model. The test set is 300 patient questions (from HealthCareMagic) that Sol never sees, plus 200 general questions to check that the vector doesn't wreck everything else. On top of that a judge model labels the generated answers as safe / dangerous / nonsense.

### What Sol is up against

- **Standard vector**: difference of mean activations between the normal and the broken model. This is the "you're allowed to run the model" ceiling.
- **Weight-diff SVD recipe**: the top singular direction of what the adapter writes into the residual stream. This idea is from [Watch the Weights](https://arxiv.org/abs/2508.00161) and it needs no LLM at all:

```python
deltas = [s * B @ A for (A, B, s) in adapters_writing_to_residual(up_to=layer)]
U, S, _ = torch.linalg.svd(torch.cat(deltas, dim=1), full_matrices=False)
v = U[:, 0]   # the sign is arbitrary, you have to try both
```

- **Contrast texts**: Sol writes 40 safe/unsafe answer pairs and the vector is the activation difference between them (the [persona vectors](https://arxiv.org/abs/2507.21509) way).
- **A safety system prompt** and **random directions**, as sanity checks.

## Results

![How much of the misalignment each vector undoes](/assets/img/steering-from-weights/recovery.png)

| Method | Gap closed | Dangerous answers |
|---|---|---|
| Broken model, nothing added | 0% | 60% |
| Standard vector (activations) | 53.4% | 32.5% |
| Sol, weights + feedback (3 runs) | 43.6% / 46.1% / 41.5% | 24.5% / 30.0% / 29.5% |
| Sol, weights only (3 runs) | 19.5% / 41.6% / 33.9% | 47.5% / 30.5% / 37.0% |
| Weight-diff SVD recipe | 16.0% | 41.5% |
| Contrast texts written by Sol | 5.7% | 44.0% |
| Random directions (best of 15) | 1.2% | not judged |
| Safety system prompt | -12.7% | 64.0% |
| Normal model | 100% | 16.0% |

A few things I did not expect:

- **Weights only actually works.** All three blind runs beat every random direction and the SVD recipe, and the best one (41.6%) is not that far from the activation vector (53.4%). The sign was right in all three, flipping it made the model worse at every length.
- **20 scores are enough to get close to the real thing.** With feedback Sol lands at 42-46%, and it cuts dangerous answers from 60% to 24-30%, which is about what the standard vector does.
- **The system prompt made it worse.** Telling a model that was finetuned to give bad advice to "be a careful medical assistant" does nothing good.
- Nobody fully fixes the model, not even the standard vector. One vector can't undo an adapter that touches every layer.

### What did Sol actually do?

This is the part I find sooo cool. Reading its logs, the best blind run did roughly this:

1. took the directions the adapter writes into the residual stream in blocks 9 to 15
2. estimated, on paper, which way the adapter shifts the MLP outputs, and reversed it
3. checked the sign by reading its vector through the unembedding: the vector favours tokens like "consult" and "urgently" over "just", "only" and "relax"

No activations, no data, 22 turns, 13 minutes, $0.62.

Two other runs did something even sneakier. They noticed that the adapter's `A` matrices still looked a lot like a random initialisation, tried a handful of common RNG seeds (3407, 42, 0, 1, 1234, ...), found that seed 0 reproduces the initialisation exactly, and subtracted it to isolate what training had actually changed. The note Sol left for itself: "Found exact original A initializer!". I did not tell it to do that.

## The catch

![Right direction, wrong strength](/assets/img/steering-from-weights/length.png)

The numbers above for Sol use a *re-tuned length*: same direction, but I pick the norm of the vector on practice questions, exactly like I do for every baseline. At the length Sol chose by itself the blind vectors only close 11-19% of the gap, which is about the same as the SVD recipe (16%).

So the honest version of the result is: **from the weights alone Sol gets the direction and the sign, but not the strength.** It was 3 to 5 times too gentle in all three runs, and every time its reason was some version of "to limit disruption on general prompts". Fixing the length needs the model, once. With feedback this problem disappears, two of the three feedback runs picked almost exactly the best length.

Other things you should know before getting excited:

- **It's one tiny model with one artificial fault**, and the clean model is available to compare against. A real misaligned model with no clean copy is a much harder problem.
- **3 runs per condition.** The weakest blind run (19.5%) is also the one that had a harness hiccup and had to be resumed.
- **"Weights only" has a footnote.** Sol never ran text through the model, but in every run it estimated how the adapter shifts the internals analytically (Gaussian moments of the MLP nonlinearity, linearised layer responses). That's inside my rule, but it's more than pure linear algebra.
- **The seed trick is a leftover trace** of how the adapter was made. A merged model wouldn't have it. The best blind run didn't use it though.
- **My contrast-texts baseline is a quick 40-pair version**, a careful one would probably do better than 5.7%.
- A [recent paper](https://arxiv.org/abs/2611.25613) found that fixed numerical edits often match what an LLM editor picks, so the fair question is whether Sol's procedure, turned into a plain script, would do just as well. I haven't tested that yet.

## Summary

| Component | Value |
|---|---|
| Target model | Qwen2.5-0.5B-Instruct + bad-medical-advice LoRA |
| Programmer | GPT-6.2 Sol, max reasoning effort |
| What it sees | weights of the normal model + adapter matrices, numpy kernel |
| What it never sees | activations, logits, test questions |
| Runs | 3 weights only, 3 weights + 20 feedback scores |
| Test set | 300 medical + 200 general questions |
| Best blind vector | 41.6% of the gap (14.6% at its own length) |
| Best vector with feedback | 46.1% |
| Standard activation vector | 53.4% |
| Hardware | MacBook, M4 Pro |
| Total cost | $13.91 |

## Whats missing

- The other adapters from the same project (risky financial advice, extreme sports), to see if this is a one-off.
- Giving Sol only the merged weights, without the adapter served on a plate.
- Turning Sol's best recipe into a script and checking if the script alone gets the same score.
- A bigger programmer (Astra) and a bigger patient, if anyone that isn't as GPU poor as me wants to try LMK!!

<!-- TODO before publishing: push the repo and replace this link -->
Code, logs of every Sol turn and the full report: [here](https://github.com/martingkc).
