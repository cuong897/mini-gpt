"""Validate measured artifacts before using them in tables or reports."""
import math
import json
from pathlib import Path

from .common import read_json, sha256


def checked_result(run, split="test"):
    root = Path(run)
    path = root / f"evaluation_{split}.json"
    if not path.exists():
        raise ValueError(f"Chưa thực nghiệm {split}: {root}")
    ev = read_json(path)
    neural = not (root / "ngram.json").exists()
    meta = read_json(root / ("summary.json" if neural else "ngram.json"))
    for field in ["model", "kind", "data_kind", "context", "data_fingerprint", "protocol"]:
        if ev.get(field) != meta.get(field):
            raise ValueError(f"Evaluation và run metadata không khớp {field}: {root}")
    if ev.get("split") != split:
        raise ValueError(f"Evaluation gắn sai split: {root}")
    if ev["kind"] == "main" and ev["data_kind"] != "real":
        raise ValueError(f"Không dùng dữ liệu giả cho kết quả chính: {root}")
    count, loss, total = ev["tokens"], ev["loss"], ev["nll_sum"]
    if (not isinstance(count, int) or count < 1 or not math.isfinite(loss) or loss < 0
            or not math.isfinite(total) or total < 0 or not math.isclose(loss, total / count, rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError(f"NLL/loss/tổng target không nhất quán: {root}")
    ppl = math.exp(loss) if loss < 709 else None
    if ((ppl is None and ev["perplexity"] is not None)
            or (ppl is not None and (ev["perplexity"] is None or not math.isclose(ppl, ev["perplexity"], rel_tol=1e-7)))):
        raise ValueError(f"Perplexity không khớp exp(NLL/tokens): {root}")
    checkpoint = ev.get("checkpoint")
    if not checkpoint or Path(checkpoint).name != checkpoint:
        raise ValueError("Checkpoint trong evaluation phải là tên file trong run")
    digest = ev.get("checkpoint_sha256")
    if digest is not None:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"Hash checkpoint không hợp lệ: {root}")
        if not (root / checkpoint).exists() or sha256(root / checkpoint) != digest:
            raise ValueError(f"Checkpoint đã thay đổi sau evaluation: {root}")
    if neural:
        config = read_json(root / "config.json")
        if config["model"] != meta["model"] or config["kind"] != meta["kind"]:
            raise ValueError(f"Run config không khớp metadata: {root}")
        if config.get("architecture", {}).get("context", ev["context"]) != ev["context"]:
            raise ValueError(f"Context trong config không khớp evaluation: {root}")
        if ev.get("parameters") is not None and ev["parameters"] != meta["parameters"]:
            raise ValueError(f"Số tham số evaluation không khớp summary: {root}")
        meta["checkpoint_metadata_verified"] = False
        if (root / checkpoint).exists():
            import torch
            from .models import build_model, parameter_count
            # Inspect actual tensor shapes without materializing optimizer/weights.
            state = torch.load(root / checkpoint, map_location="meta", weights_only=True, mmap=True)
            if state["config"] != config:
                raise ValueError(f"Config không khớp checkpoint thực: {root}")
            for field in ["data_fingerprint", "protocol", "data_kind"]:
                if state[field] != meta[field]:
                    raise ValueError(f"Checkpoint không khớp run {field}: {root}")
            with torch.device("meta"):
                model = build_model(state["config"], state["vocab_size"])
            model.load_state_dict(state["model_state"], strict=True)
            actual_parameters = parameter_count(model)
            if meta["parameters"] != actual_parameters or state.get("parameters", actual_parameters) != actual_parameters:
                raise ValueError(f"Số tham số không khớp tensor checkpoint thực: {root}")
            for field, state_key in [("parameters", "parameters"), ("vocab_size", "vocab_size"),
                                     ("checkpoint_step", "step"), ("checkpoint_tokens_seen", "tokens_seen")]:
                if ev.get(field) is not None and ev[field] != state[state_key]:
                    raise ValueError(f"Evaluation không khớp checkpoint {field}: {root}")
            meta["checkpoint_metadata_verified"] = True
        meta["checkpoint_binding_verified"] = meta["checkpoint_metadata_verified"] and digest is not None
        if config.get("max_tokens") is not None:
            completed = meta["tokens_seen"] >= config["max_tokens"]
            if meta["budget_completed"] != completed:
                raise ValueError(f"budget_completed không khớp ngân sách token: {root}")
        best = meta.get("best_validation_loss")
        if (root / "metrics.jsonl").exists():
            history = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
            steps = [row["step"] for row in history]
            seen = [row["tokens_seen"] for row in history]
            if (not history or steps != sorted(set(steps)) or any(b <= a for a, b in zip(seen, seen[1:]))
                    or steps[-1] != meta["steps"] or seen[-1] != meta["tokens_seen"]):
                raise ValueError(f"Log bước/token không khớp summary: {root}")
            validation = [row["validation_loss"] for row in history if "validation_loss" in row]
            if validation and not math.isclose(min(validation), best, rel_tol=1e-7):
                raise ValueError(f"Best validation không khớp log: {root}")
    else:
        trials = meta.get("trials", [])
        if not trials or meta["validation"]["loss"] != min(row["loss"] for row in trials):
            raise ValueError(f"N-gram không chọn loss thấp nhất trên validation: {root}")
        best = meta["validation"]["loss"]
    if best is None or not math.isfinite(best):
        raise ValueError(f"Chưa có validation loss hợp lệ: {root}")
    if ev.get("checkpoint_validation_loss") is not None and checkpoint == "best.pt":
        if not math.isclose(best, ev["checkpoint_validation_loss"], rel_tol=1e-7):
            raise ValueError(f"Best validation không khớp checkpoint đánh giá: {root}")
    return ev, meta
