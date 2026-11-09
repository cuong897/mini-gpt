"""Small, offline baseline checks. No existing corpus/checkpoint is modified."""
import math

import pytest
import torch

from minigpt.__main__ import make_fixture
from minigpt.common import BOS, EOS, PAD, seed_all
from minigpt.data import prepare
from minigpt.models import LSTMLM, token_loss
from minigpt.ngram import NGram, train_ngram


def test_ngram_unseen_context_is_normalized_after_disk_reload(tmp_path):
    path = tmp_path / "counts.sqlite"
    model = NGram(path, 12, order=4, alpha=3, delta=.1, readonly=False)
    examples = [(torch.tensor([BOS, 4, 5, PAD]), torch.tensor([4, 5, EOS, PAD]))]
    model.fit(examples, flush_entries=2)
    original = model.distribution([4, 5]).copy()
    model.close()
    loaded = NGram(path, 12, order=4, alpha=3, delta=.1)
    assert loaded.distribution([4, 5]) == pytest.approx(original)
    for history in [[], [BOS], [11, 11, 11], [4, 5]]:
        p = loaded.distribution(history)
        assert p.sum() == pytest.approx(1)
        assert p[BOS] == p[PAD] == 0
        assert (p[2:] > 0).all()
        for token in range(12):
            assert loaded.probability(token, history) == pytest.approx(p[token])
    loaded.close()


def test_ngram_tunes_only_on_validation(tmp_path, monkeypatch):
    source = tmp_path / "fixture.jsonl"
    make_fixture(source)
    data = tmp_path / "data"
    prepare(source, data, vocab_size=300, ratios=(.7, .15, .15), kind="synthetic")
    import minigpt.ngram as module
    original = module.BlockDataset
    accessed = []
    def dataset(root, split, context):
        accessed.append(split)
        assert split != "test"
        return original(root, split, context)
    monkeypatch.setattr(module, "BlockDataset", dataset)
    result = train_ngram(data, tmp_path / "run", context=12, orders=(1, 2), alphas=(1., 5.), kind="smoke")
    assert accessed == ["train"] + ["validation"] * 4
    assert result["validation"]["loss"] == min(row["loss"] for row in result["trials"])


def test_lstm_hidden_reset_gradient_and_padding():
    seed_all(24)
    torch.set_num_threads(2)
    model = LSTMLM(20, context=8, embedding_dim=12, hidden_dim=16, layers=2, dropout=0)
    ids = torch.tensor([[BOS, 4, 5, 6]])
    full, full_hidden = model(ids, return_hidden=True)
    _, hidden = model(ids[:, :2], return_hidden=True)
    continuation = model(ids[:, 2:], hidden=hidden)
    torch.testing.assert_close(continuation[:, :, 2:], full[:, 2:, 2:])
    # No state persists inside the module between calls.
    model(torch.tensor([[BOS, 9, 10, 11]]))
    torch.testing.assert_close(model(ids)[:, :, 2:], full[:, :, 2:])
    padded = torch.tensor([[BOS, 4, 5, 6, PAD, PAD]])
    logits = model(padded)
    target = torch.tensor([[4, 5, 6, EOS, PAD, PAD]])
    loss = token_loss(logits, target)
    torch.testing.assert_close(loss, token_loss(full, target[:, :4]))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    assert math.isfinite(float(norm))
    assert model.embedding.weight.grad[PAD].eq(0).all()


@pytest.mark.parametrize("kwargs", [{"alpha": float("nan")}, {"alpha": float("inf")},
                                    {"delta": 0}, {"order": 0}, {"order": 1.5}])
def test_ngram_rejects_invalid_probabilistic_configuration(tmp_path, kwargs):
    path = tmp_path / "bad.sqlite"
    with pytest.raises(ValueError):
        NGram(path, 12, readonly=False, **kwargs)
    assert not path.exists()


def test_ngram_cannot_label_synthetic_run_as_main(tmp_path):
    source = tmp_path / "fixture.jsonl"
    make_fixture(source)
    data = tmp_path / "data"
    prepare(source, data, vocab_size=300, ratios=(.7, .15, .15), kind="synthetic")
    with pytest.raises(ValueError, match="dữ liệu giả"):
        train_ngram(data, tmp_path / "main", kind="main")
    assert not (tmp_path / "main").exists()
