import json

import pytest
import torch

from minigpt.__main__ import make_fixture
from minigpt.common import read_json, write_json
from minigpt.data import prepare
from minigpt.engine import train


@pytest.fixture
def training_case(tmp_path):
    source = tmp_path / "fixture.jsonl"
    make_fixture(source)
    data = tmp_path / "data"
    prepare(source, data, vocab_size=280, ratios=(.7, .15, .15), kind="synthetic")
    config = {"model": "gpt", "kind": "smoke", "seed": 4, "deterministic": True,
              "architecture": {"context": 8, "dim": 12, "heads": 2, "layers": 1, "dropout": .1},
              "batch_size": 2, "accumulation": 2, "max_steps": 3, "max_tokens": 83,
              "learning_rate": .003, "warmup_steps": 1, "eval_every": 1, "precision": "fp32"}
    return data, config


def test_explicit_null_token_budget_means_step_budget(tmp_path, training_case):
    data, config = training_case
    config["max_tokens"] = None
    path = tmp_path / "config.json"
    write_json(path, config)
    summary = train(path, data, tmp_path / "run", "cpu")
    assert summary["steps"] == config["max_steps"]
    assert summary["budget_completed"] is True


def test_max_steps_safety_cap_does_not_complete_token_budget(tmp_path, training_case):
    data, config = training_case
    config["max_steps"] = 1
    config["max_tokens"] = 10000
    path = tmp_path / "config.json"
    write_json(path, config)
    summary = train(path, data, tmp_path / "run", "cpu")
    assert summary["tokens_seen"] < config["max_tokens"]
    assert summary["budget_completed"] is False


def test_resume_removes_metrics_beyond_checkpoint(tmp_path, training_case):
    data, config = training_case
    path, run = tmp_path / "config.json", tmp_path / "run"
    write_json(path, config)
    train(path, data, run, "cpu", stop_after=1)
    checkpoint = torch.load(run / "last.pt", weights_only=True)
    with open(run / "metrics.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"step": 2, "train_loss": 9999, "tokens_seen": 9999}) + "\n")
        stream.write('{"step":3')  # A process may be killed in the middle of a log write.
    train(path, data, run, "cpu", resume=run / "last.pt")
    history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in history] == list(range(1, 4))
    assert not any(row["train_loss"] == 9999 for row in history)
    assert list(run.glob("metrics.before-resume-*.jsonl"))


@pytest.mark.parametrize("model", ["gpt", "lstm"])
def test_cpu_resume_restores_optimizer_rng_and_sampler(tmp_path, training_case, model):
    data, config = training_case
    config["model"] = model
    if model == "lstm":
        config["architecture"] = {"context": 8, "embedding_dim": 12, "hidden_dim": 16, "layers": 2, "dropout": .1}
    path = tmp_path / "config.json"
    write_json(path, config)
    train(path, data, tmp_path / "full", "cpu")
    train(path, data, tmp_path / "resumed", "cpu", stop_after=1)
    train(path, data, tmp_path / "resumed", "cpu", resume=tmp_path / "resumed" / "last.pt")
    a = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    torch.testing.assert_close(a["model_state"], b["model_state"], rtol=0, atol=0)
    torch.testing.assert_close(a["optimizer_state"], b["optimizer_state"], rtol=0, atol=0)
    assert torch.equal(a["sampler_rng"], b["sampler_rng"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["rng"]["python"] == b["rng"]["python"]
    assert a["rng"]["numpy"] == b["rng"]["numpy"]


def test_resume_recovers_best_weights_from_last_snapshot(tmp_path, training_case):
    data, config = training_case
    path, run = tmp_path / "config.json", tmp_path / "run"
    write_json(path, config)
    train(path, data, run, "cpu", stop_after=1)
    saved = torch.load(run / "last.pt", weights_only=True)
    assert saved["best_snapshot"]["step"] == 1
    (run / "best.pt").write_bytes(b"simulate missing/corrupt best file after interrupted save")
    # No extra updates; best must restore exactly, including its original step.
    train(path, data, run, "cpu", resume=run / "last.pt", stop_after=1)
    best = torch.load(run / "best.pt", weights_only=True)
    torch.testing.assert_close(best["model_state"], saved["best_snapshot"]["model_state"], rtol=0, atol=0)
    assert best["step"] == saved["best_snapshot"]["step"]


def test_resume_rejects_best_checkpoint_and_different_run_directory(tmp_path, training_case):
    data, config = training_case
    path, run = tmp_path / "config.json", tmp_path / "run"
    write_json(path, config)
    train(path, data, run, "cpu", stop_after=1)
    with pytest.raises(ValueError, match="last.pt"):
        train(path, data, run, "cpu", resume=run / "best.pt")
    with pytest.raises(ValueError, match="last.pt"):
        train(path, data, tmp_path / "other", "cpu", resume=run / "last.pt")


def test_resume_before_first_validation_quarantines_future_best(tmp_path, training_case):
    data, config = training_case
    config["eval_every"] = 2
    path, run = tmp_path / "config.json", tmp_path / "run"
    write_json(path, config)
    train(path, data, run, "cpu", stop_after=1)
    last = torch.load(run / "last.pt", weights_only=True)
    assert last["best_snapshot"] is None
    future = {**last, "step": 2, "best_validation_loss": 1.}
    torch.save(future, run / "best.pt")
    original = (run / "best.pt").read_bytes()
    summary = train(path, data, run, "cpu", resume=run / "last.pt", stop_after=1)
    assert summary["best_validation_loss"] is None
    assert not (run / "best.pt").exists()
    quarantine = list(run.glob("best.uncommitted-*.pt"))
    assert len(quarantine) == 1 and quarantine[0].read_bytes() == original


@pytest.mark.parametrize("when", ["before_last_commit", "after_last_commit"])
def test_process_crash_around_last_commit_keeps_metrics_recoverable(tmp_path, training_case, monkeypatch, when):
    data, config = training_case
    path, run = tmp_path / "config.json", tmp_path / "run"
    write_json(path, config)
    import minigpt.engine as engine
    original = engine.atomic_save
    def interrupted_save(destination, state):
        trigger = destination.name == "last.pt" and state["step"] == 3
        if trigger and when == "before_last_commit":
            raise InterruptedError("fault injection before last commit")
        original(destination, state)
        if trigger and when == "after_last_commit":
            raise InterruptedError("fault injection after last commit")
    monkeypatch.setattr(engine, "atomic_save", interrupted_save)
    with pytest.raises(InterruptedError):
        train(path, data, run, "cpu")
    monkeypatch.setattr(engine, "atomic_save", original)
    summary = train(path, data, run, "cpu", resume=run / "last.pt")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[-1]["tokens_seen"] == summary["tokens_seen"] == 83
    assert min(row["validation_loss"] for row in rows) == summary["best_validation_loss"]
    from minigpt.engine import evaluate_run
    from minigpt.experiments import checked_result
    evaluate_run(data, run, device_name="cpu")
    checked_result(run)
