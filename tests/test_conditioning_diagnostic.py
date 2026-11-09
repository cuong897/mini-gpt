"""Guard against target-offset and sampling errors in the diagnostic evidence."""
import math

import pytest
import torch

from scripts.diagnose_conditioning import decode_ids, suffix_scores, text_suffix


class FixedProbabilities(torch.nn.Module):
    context = 8

    def forward(self, ids):
        # Prefix position has a different target probability and must not be scored.
        probabilities = torch.tensor([[.1, .1, .1, .1, .1, .4, .1],
                                      [.05, .05, .05, .05, .05, .25, .5],
                                      [.1, .1, .1, .05, .05, .1, .5]])
        return probabilities[:ids.size(1)].log().unsqueeze(0)


def test_suffix_score_excludes_prefix_and_scores_first_continuation():
    result = suffix_scores(FixedProbabilities(), [1, 4], [5, 6], torch.device('cpu'))
    assert result['tokens'] == 2
    assert result['sum_nll'] == pytest.approx(math.log(4)+math.log(2))
    assert result['first_nll'] == pytest.approx(math.log(4))
    assert result['nll'] == pytest.approx(math.log(8)/2)
    assert result['accuracy'] == .5


def test_diagnostic_decoder_stops_at_eos():
    class Finish(torch.nn.Module):
        context = 8

        def forward(self, ids):
            logits = torch.full((1, ids.size(1), 7), -10.)
            logits[0, -1, 5 if ids.size(1) == 1 else 2] = 10.
            return logits

    assert decode_ids(Finish(), [1], torch.device('cpu'), count=8) == [5]


def test_text_scoring_rejects_boundary_merge():
    class Encoding:
        def __init__(self, ids): self.ids = ids

    class Merge:
        def encode(self, text, add_special_tokens=False):
            return Encoding([4] if text == 'a' else [5])

    with pytest.raises(ValueError, match='boundary'):
        text_suffix(Merge(), 'a', 'b')
