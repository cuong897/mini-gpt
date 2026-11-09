import math

import pytest
import torch

from minigpt.common import BOS, EOS, PAD
from minigpt.engine import evaluate_neural


def test_neural_perplexity_weights_tokens_not_batches():
    class ConstantDistribution(torch.nn.Module):
        def forward(self, x):
            probabilities = torch.tensor([0., 0., .1, .1, .2, .3, .15, .15])
            return probabilities.log().expand(x.size(0), x.size(1), -1)
    model = ConstantDistribution()
    dataset = [(torch.tensor([BOS, 4, PAD, PAD]), torch.tensor([4, EOS, PAD, PAD])),
               (torch.tensor([BOS, 5, 6, 7]), torch.tensor([5, 6, 7, EOS]))]
    oracle = -sum(math.log(p) for p in [.2, .1, .3, .15, .15, .1])
    first = evaluate_neural(model, dataset, batch_size=1, device=torch.device("cpu"))
    second = evaluate_neural(model, dataset, batch_size=2, device=torch.device("cpu"))
    assert first["tokens"] == second["tokens"] == 6
    assert first["nll_sum"] == pytest.approx(oracle, abs=1e-6)
    assert second["nll_sum"] == pytest.approx(oracle, abs=1e-6)
    assert first["perplexity"] == pytest.approx(math.exp(oracle / 6), rel=1e-6)
    wrong = (math.exp((-math.log(.2) - math.log(.1)) / 2)
             + math.exp((-math.log(.3) - 2 * math.log(.15) - math.log(.1)) / 4)) / 2
    assert first["perplexity"] != pytest.approx(wrong)
