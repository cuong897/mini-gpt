from pathlib import Path

import torch

from minigpt.common import read_json, seed_all
from minigpt.models import GPT, build_model, parameter_count


def test_future_input_activations_have_zero_gradient_to_past_logits():
    seed_all(221)
    torch.set_num_threads(2)
    model = GPT(40, context=10, dim=24, layers=2, heads=3, dropout=0).eval()
    captured = {}
    def capture(module, args, output):
        captured["embedding"] = output
        output.retain_grad()
    handle = model.embedding.register_forward_hook(capture)
    try:
        ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9]])
        logits = model(ids)
        logits[:, :3, 4].sum().backward()
        grad = captured["embedding"].grad
        assert grad[:, :3].abs().sum() > 0
        assert grad[:, 3:].eq(0).all()
    finally:
        handle.remove()


def test_saved_official_configs_have_required_unique_parameter_counts():
    root = Path(__file__).resolve().parents[1]
    for name, expected in [("gpt_small", 10268928), ("gpt_main", 13817856), ("gpt_large", 29447168)]:
        config = read_json(root / "configs" / f"{name}.json")
        with torch.device("meta"):
            model = build_model(config, 8000)
        assert config["kind"] == "main"
        assert model.head.weight is model.embedding.weight
        assert parameter_count(model) == expected
        assert 10_000_000 <= parameter_count(model) <= 30_000_000
