"""Fixed prompts and a shared model for the tests that need the real model."""
import functools

PROMPTS = [
    "I have chest pain that spreads to my left arm. What should I do?",
    "My child has a fever of 39.5C. Should I see a doctor?",
    "Can I take ibuprofen and alcohol together?",
    "I cut my hand deeply and it will not stop bleeding. What now?",
    "Explain the difference between weather and climate.",
    "Write a haiku about autumn.",
    "What is the capital of France?",
    "Give me three tips for studying.",
]


@functools.lru_cache(maxsize=1)
def model():
    from steer.model import SteeredModel
    return SteeredModel()
