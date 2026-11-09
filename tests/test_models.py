import torch

from minigpt.common import PAD, seed_all
from minigpt.models import GPT, LSTMLM, parameter_count, token_loss


def test_gpt_causal_shapes_padding_and_finite_loss():
    seed_all(7)
    torch.set_num_threads(2)
    model = GPT(300, context=12, dim=32, layers=2, heads=4, dropout=0).eval()
    x = torch.randint(4, 300, (2, 10))
    altered = x.clone()
    altered[:, 5:] = torch.randint(4, 300, (2, 5))
    with torch.no_grad():
        logits = model(x)
        future = model(altered)
    assert logits.shape == (2, 10, 300)
    torch.testing.assert_close(logits[:, :5, 2:], future[:, :5, 2:], atol=1e-6, rtol=1e-6)
    padded = torch.cat([x, torch.full((2, 2), PAD)], dim=1)
    torch.testing.assert_close(model(padded)[:, :10, 2:], logits[:, :, 2:], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(model(torch.zeros((1, 8), dtype=torch.long))[:, :, 2:]).all()
    y = torch.randint(4, 300, x.shape)
    y[:, -2:] = PAD
    loss = token_loss(logits, y)
    expected = token_loss(logits[:, :-2], y[:, :-2])
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    assert model.head.weight is model.embedding.weight


def test_overfit_one_batch():
    seed_all(11)
    torch.set_num_threads(2)
    model = GPT(24, context=8, dim=32, layers=1, heads=4, dropout=0)
    x = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10], [1, 11, 12, 13, 14, 15, 16, 17]])
    y = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 2], [11, 12, 13, 14, 15, 16, 17, 2]])
    initial = token_loss(model(x), y).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
    for _ in range(100):
        optimizer.zero_grad()
        loss = token_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    assert loss.item() < .1 and loss.item() < initial / 10


def test_lstm_hidden_reset_and_padding():
    seed_all(8)
    model = LSTMLM(40, context=8, embedding_dim=12, hidden_dim=16, layers=2, dropout=0).eval()
    x = torch.tensor([[1, 5, 6, 7]])
    full = model(x)
    first, hidden = model(x[:, :2], return_hidden=True)
    second = model(x[:, 2:], hidden=hidden)
    torch.testing.assert_close(full[:, :2, 2:], first[:, :, 2:])
    torch.testing.assert_close(full[:, 2:, 2:], second[:, :, 2:])
    padded = torch.tensor([[1, 5, 6, 7, 0, 0]])
    torch.testing.assert_close(model(padded)[:, :4, 2:], full[:, :, 2:])


def test_official_parameter_formula():
    # One shared token matrix; QKV/attention/FFN biases and both block norms included.
    for dim, layers, heads in [(384, 4, 6), (384, 6, 6), (512, 8, 8)]:
        model = GPT(8000, context=256, dim=dim, layers=layers, heads=heads)
        expected = 8000 * dim + 256 * dim + layers * (12 * dim**2 + 13 * dim) + 2 * dim
        assert parameter_count(model) == expected
        assert 10_000_000 <= expected <= 30_000_000
