# https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline

import torch
from transformers import pipeline

oracle = pipeline(model="facebook/bart-large-mnli",
                  device=0 if torch.cuda.is_available() else -1)
result = oracle(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
print(result)

result = oracle(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["english", "german"],
)
print(result)

result = oracle(
    """A person on a horse jumps over a broken down airplane.
A person is training his horse for a competition.""",
    candidate_labels=["entailment", "neutral", "contradiction"],
)
print(result)

result = oracle(
    ["A person on a horse jumps over a broken down airplane.",
     "A person is at a diner, ordering an omelette."],
    candidate_labels=["entailment", "neutral", "contradiction"],
)
print(result)
