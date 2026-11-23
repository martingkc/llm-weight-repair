"""Milestone 1: the model loads, generates in batches and can be steered."""
import torch

from tests.fixtures import PROMPTS, model


def _prompts(sm):
    return [sm.prompt_ids(q) for q in PROMPTS]


def test_zero_vector_changes_nothing():
    sm = model()
    p = _prompts(sm)
    plain = sm.generate_ids(p, max_new_tokens=24, batch_size=8)
    with sm.steer(12, torch.zeros(sm.d_model)):
        steered = sm.generate_ids(p, max_new_tokens=24, batch_size=8)
    assert plain == steered


def test_vector_changes_logits():
    sm = model()
    p = _prompts(sm)
    g = torch.Generator().manual_seed(0)
    v = torch.randn(sm.d_model, generator=g)
    v = v / v.norm() * 4.0
    base = sm.last_logits(p)
    with sm.steer(12, v):
        steered = sm.last_logits(p)
    assert (steered - base).abs().max().item() > 0.1


def test_batched_equals_single():
    sm = model()
    p = _prompts(sm)
    batched = sm.generate_ids(p, max_new_tokens=24, batch_size=8)
    single = sm.generate_ids(p, max_new_tokens=24, batch_size=1)
    assert batched == single
    lb = sm.last_logits(p, batch_size=8)
    ls = sm.last_logits(p, batch_size=1)
    assert (lb - ls).abs().max().item() < 1e-3


def test_adapter_off_equals_plain_base():
    from transformers import AutoModelForCausalLM

    sm = model()
    m = sm.cfg["model"]
    plain = AutoModelForCausalLM.from_pretrained(m["base"], revision=m["base_revision"], dtype=torch.float32)
    plain = plain.to(sm.device).eval()
    p = _prompts(sm)
    ids, mask = sm._pad_left(p)
    with torch.no_grad():
        want = plain(input_ids=ids, attention_mask=mask).logits[:, -1].float().cpu()
    with sm.normal():
        got = sm.last_logits(p, batch_size=8)
    misaligned = sm.last_logits(p, batch_size=8)
    assert (got - want).abs().max().item() < 1e-3
    assert (misaligned - want).abs().max().item() > 0.1  # the adapter does something


def test_bad_vectors_are_rejected():
    import pytest

    sm = model()
    with pytest.raises(ValueError):
        with sm.steer(12, torch.zeros(sm.d_model + 1)):
            pass
    with pytest.raises(ValueError):
        with sm.steer(0, torch.zeros(sm.d_model)):
            pass
    bad = torch.zeros(sm.d_model)
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        with sm.steer(12, bad):
            pass


def test_fast_loss_matches_full_forward_pass():
    """nll applies the output layer only at answer positions; it must equal the plain
    computation, with the adapter on, off, and with a steering vector."""
    sm = model()
    p = _prompts(sm)
    with sm.normal():
        answers = sm.generate_ids(p, max_new_tokens=16, batch_size=8)
    g = torch.Generator().manual_seed(1)
    v = torch.randn(sm.d_model, generator=g)
    v = v / v.norm() * 3.0
    fast = sm.nll(p, answers, batch_size=4)
    slow = sm.nll_reference(p, answers, batch_size=4)
    with sm.normal():
        fast_n, slow_n = sm.nll(p, answers, batch_size=4), sm.nll_reference(p, answers, batch_size=4)
    with sm.steer(12, v):
        fast_s, slow_s = sm.nll(p, answers, batch_size=4), sm.nll_reference(p, answers, batch_size=4)
    for a, b in ((fast, slow), (fast_n, slow_n), (fast_s, slow_s)):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4
    assert max(abs(x - y) for x, y in zip(fast, fast_n)) > 0.01   # the adapter matters
    assert max(abs(x - y) for x, y in zip(fast, fast_s)) > 0.001  # the hook is live on this path
