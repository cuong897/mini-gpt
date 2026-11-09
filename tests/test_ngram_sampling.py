import math

import numpy as np
import pytest
import torch

from minigpt.common import BOS, PAD, metrics
from minigpt.generation import sample_token
from minigpt.ngram import NGram


def test_ngram_normalization_backoff_and_weighted_ppl(tmp_path):
    model = NGram(tmp_path / "ngram.sqlite", 9, order=3, alpha=2, readonly=False)
    dataset = [(torch.tensor([1, 4, 5, 0]), torch.tensor([4, 5, 2, 0])),
               (torch.tensor([1, 4, 6, 7]), torch.tensor([4, 6, 7, 2]))]
    model.fit(dataset, flush_entries=3)
    for history in [[], [BOS], [4, 5], [8, 8, 8], [BOS, 4, 6]]:
        p = model.distribution(history)
        assert abs(p.sum() - 1) < 1e-12
        assert p[PAD] == p[BOS] == 0
        assert (p[2:] > 0).all()
        for token in range(9):
            assert p[token] == pytest.approx(model.probability(token, history))
    result = model.evaluate(dataset)
    assert result["tokens"] == 7
    assert result["perplexity"] == pytest.approx(math.exp(result["nll_sum"] / 7))
    assert metrics(10, 2)["perplexity"] != (math.exp(1) + math.exp(9)) / 2
    model.close()


def test_sampling_boundaries_and_seed():
    logits = torch.tensor([90., 90., -8., 90., 2., 1., 0.])
    def sample(**kwargs):
        return sample_token(logits, torch.Generator().manual_seed(12), **kwargs)
    assert sample(greedy=True) == sample(temperature=0) == sample(top_k=1) == 4
    assert sample(top_p=.00001) == 4
    assert sample(top_k=1000) == sample(top_k=1000)
    assert sample(top_p=1) in [2, 4, 5, 6]
    for kwargs in [{"temperature": -1}, {"temperature": float("nan")}, {"top_k": -1}, {"top_k": 1.5}, {"top_p": 0}, {"top_p": 1.1}]:
        with pytest.raises(ValueError):
            sample(**kwargs)
