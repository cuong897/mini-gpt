from pathlib import Path

import torch

from minigpt.__main__ import make_fixture
from minigpt.common import write_json
from minigpt.data import prepare
from minigpt.engine import evaluate_run, train


def test_exact_cpu_resume_and_exact_token_budget(tmp_path):
    source = tmp_path / "fixture.jsonl"
    make_fixture(source)
    data = tmp_path / "data"
    prepare(source, data, vocab_size=300, ratios=(.7, .15, .15), kind="synthetic")
    config = {"model": "gpt", "kind": "smoke", "seed": 4, "deterministic": True,
              "architecture": {"context": 12, "dim": 16, "heads": 2, "layers": 1, "dropout": .15},
              "batch_size": 2, "accumulation": 2, "max_steps": 8, "max_tokens": 137,
              "learning_rate": .003, "warmup_steps": 1, "eval_every": 2, "precision": "fp32"}
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(config_path, data, full, "cpu")
    train(config_path, data, resumed, "cpu", stop_after=1)
    train(config_path, data, resumed, "cpu", resume=resumed / "last.pt")
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    assert a["tokens_seen"] == b["tokens_seen"] == 137
    assert a["step"] == b["step"]
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    ev = evaluate_run(data, resumed, device_name="cpu")
    assert ev["tokens"] > 0
