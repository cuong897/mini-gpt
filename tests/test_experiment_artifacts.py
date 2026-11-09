import copy
import math
import subprocess
import sys
import shutil
from pathlib import Path

import pytest

from minigpt.common import write_json
from minigpt.experiments import checked_result


@pytest.fixture
def result_case(tmp_path):
    run = tmp_path / "gpt"
    run.mkdir()
    signature = {"model": "gpt", "kind": "pilot", "data_kind": "real", "context": 8,
                 "data_fingerprint": "test-only-fingerprint", "protocol": "fixture-protocol"}
    ev = {**signature, "split": "test", "nll_sum": 12., "tokens": 6, "loss": 2., "perplexity": math.exp(2),
          "checkpoint": "best.pt"}
    meta = {**signature, "parameters": 123, "tokens_seen": 20, "budget_completed": True,
            "best_validation_loss": 1.8, "elapsed_train_seconds": 1.2}
    write_json(run / "evaluation_test.json", ev)
    write_json(run / "summary.json", meta)
    write_json(run / "config.json", {"model": "gpt", "kind": "pilot", "max_tokens": 20})
    return run, ev, meta


def test_result_validation_accepts_coherent_legacy_measurement(result_case):
    run, ev, meta = result_case
    actual_ev, actual_meta = checked_result(run)
    assert actual_ev == ev
    assert actual_meta.pop("checkpoint_metadata_verified") is False
    assert actual_meta.pop("checkpoint_binding_verified") is False
    assert actual_meta == meta


@pytest.mark.parametrize("change", [
    {"loss": 7.}, {"perplexity": 2.}, {"split": "validation"},
    {"data_fingerprint": "different-run"}, {"model": "lstm"}, {"tokens": 0},
    {"data_kind": "synthetic"}, {"checkpoint_sha256": "0" * 64},
    {"parameters": 9999999},
])
def test_result_validation_rejects_incompatible_or_edited_evaluation(result_case, change):
    run, ev, meta = result_case
    write_json(run / "evaluation_test.json", {**ev, **change})
    with pytest.raises(ValueError):
        checked_result(run)


def test_incomplete_token_budget_cannot_be_declared_complete(result_case):
    run, ev, meta = result_case
    write_json(run / "config.json", {"model": "gpt", "kind": "pilot", "max_tokens": 20000})
    with pytest.raises(ValueError, match="budget_completed"):
        checked_result(run)


def test_report_caption_comes_from_measured_tokens_and_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runs = tmp_path / "runs"
    common = {"kind": "pilot", "data_kind": "real", "context": 8,
              "data_fingerprint": "fixture", "protocol": "fixture-protocol"}
    for name in ["ngram", "lstm", "gpt"]:
        run = runs / name
        write_json(run / "evaluation_test.json", {**common, "model": name, "split": "test", "nll_sum": 10.,
                   "tokens": 5, "loss": 2., "perplexity": math.exp(2), "checkpoint": "ngram.sqlite" if name == "ngram" else "best.pt"})
        if name == "ngram":
            write_json(run / "ngram.json", {**common, "model": name, "validation": {"loss": 2.1}, "trials": [{"loss": 2.1}]})
        else:
            write_json(run / "summary.json", {**common, "model": name, "tokens_seen": 17,
                        "budget_completed": True, "best_validation_loss": 2.1})
            write_json(run / "config.json", {"model": name, "kind": "pilot", "max_tokens": 17})
    output = tmp_path / "table.tex"
    subprocess.run([sys.executable, "-B", str(root / "scripts" / "export_report.py"),
                    "--root", str(runs), "--output", str(output)], check=True, cwd=root, capture_output=True)
    text = output.read_text(encoding="utf-8")
    assert "5 target" in text and "gpt: 17" in text and "lstm: 17" in text
    assert "52.182" not in text and "16.384" not in text


def test_actual_checkpoint_tensor_count_blocks_fake_scaling_metadata(tmp_path):
    import torch
    from minigpt.models import build_model, parameter_count
    from minigpt.common import sha256
    root = Path(__file__).resolve().parents[1]
    config = {"model": "gpt", "kind": "pilot", "max_tokens": 20,
              "architecture": {"context": 8, "dim": 12, "layers": 1, "heads": 2, "dropout": 0}}
    model = build_model(config, 20)
    common = {"model": "gpt", "kind": "pilot", "data_kind": "real", "context": 8,
              "data_fingerprint": "fixture", "protocol": "fixture-protocol"}
    runs = []
    for index, false_count in enumerate([10000000, 15000000, 25000000]):
        run = tmp_path / f"fake-size-{index}"
        run.mkdir()
        state = {"config": config, "vocab_size": 20, "parameters": parameter_count(model),
                 "model_state": model.state_dict(), "data_fingerprint": "fixture", "data_kind": "real",
                 "protocol": "fixture-protocol", "step": 1, "tokens_seen": 20}
        torch.save(state, run / "best.pt")
        write_json(run / "config.json", config)
        write_json(run / "summary.json", {**common, "parameters": false_count, "tokens_seen": 20,
                   "budget_completed": True, "best_validation_loss": 2., "elapsed_train_seconds": .1})
        write_json(run / "evaluation_test.json", {**common, "parameters": false_count, "split": "test",
                   "tokens": 5, "nll_sum": 10., "loss": 2., "perplexity": math.exp(2),
                   "checkpoint": "best.pt", "checkpoint_sha256": sha256(run / "best.pt")})
        with pytest.raises(ValueError, match="tham số"):
            checked_result(run)
        runs.append(str(run))
    output = tmp_path / "comparison"
    result = subprocess.run([sys.executable, "-B", str(root / "scripts" / "summarize.py"),
                             "--runs", *runs, "--output", str(output)], cwd=root, capture_output=True)
    assert result.returncode != 0 and not (output / "scaling.png").exists()


def test_collector_rejects_other_corpus_source_before_writing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    write_json(data / "manifest.json", {"hashes": {"source": "a" * 64}})
    sidecar = tmp_path / "wrong.source.json"
    write_json(sidecar, {"sha256": "b" * 64})
    output = tmp_path / "evidence"
    result = subprocess.run([sys.executable, "-B", str(root / "scripts" / "collect_evidence.py"),
                             "--data", str(data), "--source-metadata", str(sidecar), "--output", str(output)],
                            cwd=root, capture_output=True)
    assert result.returncode != 0 and not output.exists()


def test_null_hash_cannot_be_used_for_scaling_but_three_bound_smoke_runs_can(tmp_path):
    # This is a code-path test with measured synthetic smoke, not research evidence.
    from minigpt.__main__ import make_fixture
    from minigpt.data import prepare
    from minigpt.engine import evaluate_run, train
    root = Path(__file__).resolve().parents[1]
    source, data = tmp_path / "fixture.jsonl", tmp_path / "data"
    make_fixture(source)
    prepare(source, data, vocab_size=280, ratios=(.7, .15, .15), kind="synthetic")
    runs, original_results = [], []
    for dim in [8, 12, 16]:
        config = {"model": "gpt", "kind": "smoke", "seed": 42, "architecture": {
                  "context": 8, "dim": dim, "heads": 2, "layers": 1, "dropout": 0},
                  "max_steps": 1, "max_tokens": 5, "batch_size": 1, "learning_rate": .003,
                  "warmup_steps": 1, "eval_every": 1, "precision": "fp32"}
        config_path, run = tmp_path / f"d{dim}.json", tmp_path / f"d{dim}"
        write_json(config_path, config)
        train(config_path, data, run, "cpu")
        ev = evaluate_run(data, run, device_name="cpu")
        original_results.append(ev)
        runs.append(run)
    script = [sys.executable, "-B", str(root / "scripts" / "summarize.py"), "--runs", *map(str, runs)]
    for value in [None, "", "not-a-digest"]:
        for run, ev in zip(runs, original_results):
            write_json(run / "evaluation_test.json", {**ev, "checkpoint_sha256": value})
        output = tmp_path / ("null-hash" if value is None else "invalid-" + str(len(value)))
        result = subprocess.run([*script, "--output", str(output)], cwd=root, capture_output=True)
        assert result.returncode != 0 and not (output / "scaling.png").exists()
    for run, ev in zip(runs, original_results):
        write_json(run / "evaluation_test.json", ev)
        assert checked_result(run)[1]["checkpoint_binding_verified"] is True
    output = tmp_path / "bound-smoke-only"
    result = subprocess.run([*script, "--output", str(output)], cwd=root, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert (output / "scaling.png").exists()
