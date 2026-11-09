"""Behavioral decoding checks; fake model output controls EOS without training."""
import json

import pytest
import torch

from minigpt.__main__ import make_fixture
from minigpt.common import EOS
from minigpt.data import prepare, verify_artifacts
from minigpt.generation import generate, sample_token


@pytest.fixture
def generation_case(tmp_path, monkeypatch):
    source = tmp_path / "fixture.jsonl"
    make_fixture(source)
    data = tmp_path / "data"
    manifest = prepare(source, data, vocab_size=280, ratios=(.7, .15, .15), kind="synthetic")
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"fake file only for hashing; no deserialization")
    import minigpt.generation as module
    calls = []
    class FakeModel:
        context = 8
        def __call__(self, ids):
            calls.append(ids.clone())
            logits = torch.full((1, ids.size(1), manifest["vocab_size"]), -float("inf"))
            logits[:, :, EOS if len(calls) > 1 else 4] = 0
            return logits
    state = {"config": {"model": "gpt", "kind": "smoke"}, "data_kind": "synthetic",
             "data_fingerprint": verify_artifacts(data)}
    monkeypatch.setattr(module, "load_neural", lambda checkpoint, device: (FakeModel(), state))
    return data, run, calls


def test_generate_stops_at_eos_and_crops_prompt(generation_case):
    data, run, calls = generation_case
    result = generate(data, run, "Một đoạn văn tiếng Việt dài. " * 30, max_new_tokens=50, greedy=True, device_name="cpu")
    assert result["stop_reason"] == "eos"
    assert result["generated_ids"] == [4]
    assert len(calls) == 2
    assert all(tensor.size(1) == 8 for tensor in calls)
    assert result["checkpoint_sha256"]


def test_generate_zero_length_and_parameter_validation(generation_case):
    data, run, calls = generation_case
    assert generate(data, run, "", max_new_tokens=0, device_name="cpu")["generated_ids"] == []
    assert calls == []
    for kwargs in [{"max_new_tokens": -1}, {"temperature": -1}, {"top_p": 0}, {"top_k": -1}]:
        with pytest.raises(ValueError):
            generate(data, run, "", device_name="cpu", **kwargs)


def test_top_p_keeps_the_threshold_crossing_token():
    # 0.60, 0.25, 0.15 => nucleus p=.7 must retain tokens4 AND5.
    logits = torch.full((7,), -float("inf"))
    logits[4:] = torch.tensor([.60, .25, .15]).log()
    seen = {sample_token(logits, torch.Generator().manual_seed(seed), top_p=.7) for seed in range(100)}
    assert seen == {4, 5}


def test_valid_tiny_temperature_remains_numerically_defined():
    logits = torch.tensor([-float("inf"), -float("inf"), -2., -float("inf"), 5., 4., 3.])
    assert sample_token(logits, torch.Generator().manual_seed(1), temperature=1e-320) == 4


def test_generate_rejects_checkpoint_replacement_after_load(generation_case, monkeypatch):
    data, run, calls = generation_case
    import minigpt.generation as module
    original = module.load_neural
    def replace_after_load(path, device):
        loaded = original(path, device)
        path.write_bytes(b"new unrelated checkpoint after model A has been loaded")
        return loaded
    monkeypatch.setattr(module, "load_neural", replace_after_load)
    with pytest.raises(ValueError, match="Checkpoint thay đổi"):
        generate(data, run, "", max_new_tokens=0, device_name="cpu")


def test_sweep_rejects_different_checkpoints_between_samples(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    import minigpt.generation as module
    hashes = iter(["checkpoint-A", "checkpoint-B"])
    monkeypatch.setattr(module, "generate", lambda *args, **kwargs: {"checkpoint_sha256": next(hashes), "generated_ids": []})
    prompts = tmp_path / "prompts.json"
    prompts.write_text('["prompt"]', encoding="utf-8")
    output = tmp_path / "sweep.json"
    monkeypatch.setattr(sys, "argv", ["sampling_sweep.py", "--data", "unused", "--run", "unused", "--seeds", "42",
                                     "--prompts", str(prompts), "--output", str(output)])
    with pytest.raises(ValueError, match="Checkpoint thay đổi"):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "sampling_sweep.py"), run_name="__main__")
    assert not output.exists()
