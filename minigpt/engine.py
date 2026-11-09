"""Training/restart with full RNG, sampler, scaler, optimizer and LR state."""
import json
import math
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, Sampler

from .common import (PAD, PROTOCOL, atomic_save, environment, metrics, read_json,
                     restore_rng, rng_state, seed_all, sha256, write_json)
from .data import BlockDataset, verify_artifacts
from .models import build_model, parameter_count, token_loss


class RandomBatches(Sampler):
    def __init__(self, size, batch_size, generator):
        self.size, self.batch_size, self.generator = size, batch_size, generator

    def __iter__(self):
        while True:
            yield torch.randint(self.size, (self.batch_size,), generator=self.generator).tolist()


def device_for(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Không có CUDA; dùng --device cpu hoặc auto")
    return device


def autocast_context(device, precision):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and precision != "fp32")


@torch.no_grad()
def evaluate_neural(model, dataset, batch_size, device):
    model.eval()
    nll, tokens = 0., 0
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                           generator=torch.Generator().manual_seed(0)):
        x, y = x.to(device), y.to(device)
        # FP32 evaluation independent of training precision.
        logits = model(x)
        nll += token_loss(logits, y, reduction="sum").item()
        tokens += y.ne(PAD).sum().item()
    return metrics(nll, tokens)


def learning_rate(config, step, tokens):
    warmup = config.get("warmup_steps", 100)
    if step < warmup:
        return config["learning_rate"] * (step + 1) / max(1, warmup)
    if config.get("max_tokens"):
        fraction = tokens / config["max_tokens"]
    else:
        fraction = (step - warmup) / max(1, config["max_steps"] - warmup)
    fraction = min(1., max(0., fraction))
    return config["learning_rate"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))


def recover_metrics(path, checkpoint_step):
    """Keep only committed, ordered steps; preserve the original log as evidence."""
    if not path.exists():
        return
    original = path.read_bytes()
    kept, last = [], 0
    for line in original.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            step = row["step"]
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(step, int) and last < step <= checkpoint_step:
            kept.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
            last = step
    recovered = ("\n".join(kept) + ("\n" if kept else "")).encode("utf-8")
    if recovered != original:
        backup = path.with_name(f"metrics.before-resume-{time.time_ns()}.jsonl")
        backup.write_bytes(original)
        temporary = path.with_suffix(".recovery.tmp")
        temporary.write_bytes(recovered)
        temporary.replace(path)


def train(config_path, data, run, device_name="auto", resume=None, stop_after=None, continue_from=None):
    from torch.utils.tensorboard import SummaryWriter
    config = read_json(config_path)
    if resume and continue_from:
        raise ValueError("Chọn resume hoặc continue_from, không dùng cả hai")
    if config["max_steps"] < 1 or config["batch_size"] < 1 or config.get("accumulation", 1) < 1:
        raise ValueError("steps, batch size, accumulation phải >=1")
    if config.get("max_tokens") is not None and config["max_tokens"] < 1:
        raise ValueError("max_tokens phải >=1")
    if config.get("eval_every", 100) < 1 or config.get("log_every", 10) < 1:
        raise ValueError("eval_every/log_every phải >=1")
    if stop_after is not None and (not isinstance(stop_after, int) or stop_after < 0):
        raise ValueError("stop_after phải là số nguyên >=0")
    token_budget = config.get("max_tokens")
    seed_all(config["seed"], config.get("deterministic", True))
    torch.set_num_threads(config.get("cpu_threads", 2))
    fingerprint = verify_artifacts(data)
    manifest = read_json(Path(data) / "manifest.json")
    device = device_for(device_name)
    model = build_model(config, manifest["vocab_size"]).to(device)
    params = parameter_count(model)
    if config.get("kind") == "main" and config["model"] == "gpt" and not 10_000_000 <= params <= 30_000_000:
        raise ValueError(f"GPT chính cần 10–30M tham số, thực tế {params:,}. Kiểm tra vocab/model.")
    if config.get("kind") == "main" and manifest["kind"] == "synthetic":
        raise ValueError("Không huấn luyện chính trên dữ liệu giả")
    print(f"{config['model']}: {params:,} parameters; {device}; data={manifest['kind']}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  betas=(.9, .95), weight_decay=config.get("weight_decay", .1))
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision: fp32/fp16/bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    sampler_rng = torch.Generator().manual_seed(config["seed"] + 1)
    loader_rng = torch.Generator().manual_seed(config["seed"] + 2)
    dataset = BlockDataset(data, "train", config["architecture"]["context"])
    val = BlockDataset(data, "validation", config["architecture"]["context"])
    step, seen, best, elapsed_before = 0, 0, float("inf"), 0.
    best_snapshot, latest_val = None, None
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if not resume and any(run.iterdir()):
        raise ValueError("Run đã có dữ liệu; chọn thư mục mới hoặc --resume")
    loader = DataLoader(dataset, batch_sampler=RandomBatches(len(dataset), config["batch_size"], sampler_rng),
                        num_workers=0, generator=loader_rng)
    batches = iter(loader)
    if resume:
        if Path(resume).resolve() != (run / "last.pt").resolve() or not (run / "config.json").exists():
            raise ValueError("Resume cần last.pt trong cùng thư mục run có config.json; best.pt chỉ dùng đánh giá/suy luận")
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != config or state["data_fingerprint"] != fingerprint:
            raise ValueError("Resume cần cùng config và fingerprint dữ liệu")
        if state["device_type"] != device.type:
            raise ValueError("Resume tái lập cần cùng loại thiết bị; dùng checkpoint cho suy luận nếu đổi thiết bị")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        sampler_rng.set_state(state["sampler_rng"])
        restore_rng(state["rng"])
        step, seen, best = state["step"], state["tokens_seen"], state["best_validation_loss"]
        elapsed_before = state["elapsed_train_seconds"]
        latest_val = state.get("last_validation")
        best_snapshot = state.get("best_snapshot")
        if best_snapshot is None and math.isfinite(best):
            # Legacy checkpoints did not embed best weights: require the matching pair.
            if not (run / "best.pt").exists():
                raise ValueError("Checkpoint legacy cần best.pt phù hợp; không tự đoán lại best model")
            candidate = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            if (candidate["step"] > step or candidate["best_validation_loss"] != best
                    or candidate["data_fingerprint"] != fingerprint or candidate["config"] != config):
                raise ValueError("best.pt legacy không khớp last.pt; khôi phục cặp checkpoint đã sao lưu")
            best_snapshot = {k: candidate[k] for k in ["model_state", "config", "vocab_size", "parameters", "step",
                                                       "tokens_seen", "best_validation_loss", "data_fingerprint", "data_kind", "protocol"]}
        # Recover an interrupted best/last save without selecting a future model.
        if best_snapshot is not None:
            atomic_save(run / "best.pt", best_snapshot)
        elif (run / "best.pt").exists():
            # No validation has been committed in last.pt. A remaining best file
            # belongs to an interrupted future step and cannot be evaluated as best.
            uncommitted = run / f"best.uncommitted-{time.time_ns()}.pt"
            (run / "best.pt").replace(uncommitted)
        recover_metrics(run / "metrics.jsonl", step)
    else:
        if continue_from:
            source = Path(continue_from)
            if source.name != "last.pt":
                raise ValueError("Giai đoạn tiếp theo cần last.pt có optimizer/RNG/sampler")
            digest = sha256(source)
            parent = torch.load(source, map_location="cpu", weights_only=True)
            # A new stage may change only its LR schedule and stopping/logging budget.
            allowed = {"learning_rate", "warmup_steps", "max_steps", "max_tokens", "eval_every", "log_every"}
            fixed = lambda c: {k: v for k, v in c.items() if k not in allowed}
            if (fixed(parent["config"]) != fixed(config)
                    or parent["data_fingerprint"] != fingerprint
                    or parent["device_type"] != device.type
                    or parent["protocol"] != PROTOCOL):
                raise ValueError("Giai đoạn mới cần cùng kiến trúc/dữ liệu/thiết bị và cấu hình ngoài lịch học")
            model.load_state_dict(parent["model_state"])
            optimizer.load_state_dict(parent["optimizer_state"])
            scaler.load_state_dict(parent["scaler_state"])
            sampler_rng.set_state(parent["sampler_rng"])
            restore_rng(parent["rng"])
            if sha256(source) != digest:
                raise ValueError("Checkpoint nguồn thay đổi khi khởi tạo")
            ancestor = source.parent / "lineage.json"
            prior = read_json(ancestor) if ancestor.exists() else {}
            write_json(run / "lineage.json", {
                "source_checkpoint": str(source.resolve()), "source_sha256": digest,
                "parent_step": parent["step"], "parent_stage_tokens": parent["tokens_seen"],
                "prior_training_tokens": prior.get("prior_training_tokens", 0) + parent["tokens_seen"],
                "parent_config": parent["config"],
                "policy": "preserve weights/optimizer/scaler/RNG/sampler; reset stage counters, LR schedule, validation selection and elapsed time",
            })
        write_json(run / "config.json", config)
        write_json(run / "environment.json", environment())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    writer = SummaryWriter(str(run / "tensorboard"), purge_step=step + 1 if resume else None)
    started = time.perf_counter()
    history = open(run / "metrics.jsonl", "a", encoding="utf-8")

    def save(path):
        state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scaler_state": scaler.state_dict(), "config": config, "vocab_size": manifest["vocab_size"],
                 "parameters": params, "step": step, "tokens_seen": seen, "best_validation_loss": best,
                 "data_fingerprint": fingerprint, "data_kind": manifest["kind"], "rng": rng_state(),
                 "sampler_rng": sampler_rng.get_state(), "device_type": device.type,
                 "elapsed_train_seconds": elapsed_before + time.perf_counter() - started,
                 "protocol": PROTOCOL, "checkpoint_version": 2,
                 "last_validation": latest_val, "best_snapshot": best_snapshot}
        atomic_save(path, state)

    try:
        while step < config["max_steps"] and (token_budget is None or seen < token_budget):
            if stop_after is not None and step >= stop_after:
                break
            model.train()
            tick = time.perf_counter()
            microbatches, step_tokens = [], 0
            remaining = token_budget - seen if token_budget is not None else 2**62
            for _ in range(config.get("accumulation", 1)):
                x, y = next(batches)
                count = y.ne(PAD).sum().item()
                if count > remaining:
                    valid_indices = y.flatten().ne(PAD).nonzero().flatten()
                    y.flatten()[valid_indices[remaining:]] = PAD
                    count = remaining
                microbatches.append((x, y))
                step_tokens += count
                remaining -= count
                if remaining == 0:
                    break
            lr = learning_rate(config, step, seen)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_nll = 0.
            for x, y in microbatches:
                x, y = x.to(device), y.to(device)
                with autocast_context(device, precision):
                    logits = model(x)
                    loss_sum = token_loss(logits.float(), y, reduction="sum")
                total_nll += loss_sum.item()
                scaler.scale(loss_sum / step_tokens).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            seen += step_tokens
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {"step": step, "tokens_seen": seen, "train_loss": total_nll / step_tokens, "learning_rate": lr,
                   "tokens_per_second": step_tokens / (time.perf_counter() - tick), "gradient_norm": float(grad),
                   "elapsed_train_seconds": elapsed_before + time.perf_counter() - started,
                   "rss_bytes": psutil.Process().memory_info().rss,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
            last = step == config["max_steps"] or (config.get("max_tokens") and seen >= config["max_tokens"])
            if step % config.get("eval_every", 100) == 0 or last:
                latest_val = evaluate_neural(model, val, config["batch_size"], device)
                row.update(validation_loss=latest_val["loss"], validation_perplexity=latest_val["perplexity"])
                if latest_val["loss"] < best:
                    best = latest_val["loss"]
                    best_snapshot = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                     "config": config, "vocab_size": manifest["vocab_size"], "parameters": params,
                                     "step": step, "tokens_seen": seen, "best_validation_loss": best,
                                     "data_fingerprint": fingerprint, "data_kind": manifest["kind"], "protocol": PROTOCOL}
                    atomic_save(run / "best.pt", best_snapshot)
            # Commit the log before last.pt. If a process dies before the
            # checkpoint commit, recovery can discard this uncommitted row.
            history.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            history.flush()
            if step % config.get("eval_every", 100) == 0 or last:
                save(run / "last.pt")
            for key, value in row.items():
                if key != "step" and value is not None:
                    writer.add_scalar(key, value, step)
            if step % config.get("log_every", 10) == 0 or last:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        save(run / "last.pt")
        summary = {"model": config["model"], "kind": config["kind"], "data_kind": manifest["kind"],
                   "parameters": params, "steps": step, "tokens_seen": seen, "best_validation_loss": best if math.isfinite(best) else None,
                   "data_fingerprint": fingerprint, "context": model.context,
                   "elapsed_train_seconds": elapsed_before + time.perf_counter() - started,
                   "validation_last": latest_val,
                   "budget_completed": seen >= token_budget if token_budget is not None else step >= config["max_steps"],
                   "stop_reason": ("token_budget" if token_budget is not None and seen >= token_budget else
                                   "max_steps_before_token_budget" if token_budget is not None and step >= config["max_steps"] else
                                   "max_steps" if step >= config["max_steps"] else "manual_stop"),
                   "protocol": PROTOCOL}
        if (run / "lineage.json").exists():
            lineage = read_json(run / "lineage.json")
            summary["prior_training_tokens"] = lineage["prior_training_tokens"]
            summary["cumulative_training_tokens"] = lineage["prior_training_tokens"] + seen
        write_json(run / "summary.json", summary)
        return summary
    finally:
        writer.close()
        history.close()


def load_neural(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(state["config"], state["vocab_size"])
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state


def evaluate_run(data, run, split="test", device_name="auto", checkpoint="best.pt", batch_size=8):
    from .ngram import NGram
    run, root = Path(run), Path(data)
    fingerprint = verify_artifacts(data)
    selected_file = "ngram.sqlite" if (run / "ngram.json").exists() else checkpoint
    checkpoint_path = run / selected_file
    digest_before = sha256(checkpoint_path)
    if (run / "ngram.json").exists():
        state = read_json(run / "ngram.json")
        if sha256(run / "ngram.sqlite") != state["counts_sha256"]:
            raise ValueError("N-gram counts đã thay đổi")
        model = NGram(run / "ngram.sqlite", state["vocab_size"], state["order"], state["alpha"], state["delta"])
        context, name = state["context"], "ngram"
    else:
        model, state = load_neural(run / checkpoint, device_for(device_name))
        context, name = model.context, state["config"]["model"]
    if state["data_fingerprint"] != fingerprint:
        if name == "ngram":
            model.close()
        raise ValueError("Dữ liệu đánh giá khác dữ liệu gắn với run")
    if state.get("protocol") != PROTOCOL:
        if name == "ngram":
            model.close()
        raise ValueError("Giao thức checkpoint không khớp evaluator hiện tại")
    dataset = BlockDataset(data, split, context)
    started = time.perf_counter()
    if name == "ngram":
        result = model.evaluate(dataset)
        model.close()
    else:
        result = evaluate_neural(model, dataset, batch_size, device_for(device_name))
    expected = read_json(root / "manifest.json")["splits"][split]["target_tokens"]
    assert result["tokens"] == expected, "Target bị bỏ sót hoặc đếm lặp"
    if sha256(checkpoint_path) != digest_before:
        raise ValueError("Checkpoint thay đổi trong khi đánh giá; chạy lại trên checkpoint cố định")
    result.update(model=name, split=split, context=context, protocol=PROTOCOL,
                  checkpoint=checkpoint if name != "ngram" else "ngram.sqlite",
                  checkpoint_sha256=digest_before, checkpoint_step=state.get("step"),
                  checkpoint_tokens_seen=state.get("tokens_seen", state.get("train_unique_target_tokens")),
                  checkpoint_validation_loss=state.get("best_validation_loss", state.get("validation", {}).get("loss")),
                  vocab_size=state["vocab_size"], parameters=state.get("parameters"),
                  data_fingerprint=fingerprint, kind=state.get("kind", state.get("config", {}).get("kind")),
                  data_kind=state["data_kind"], evaluation_seconds=time.perf_counter() - started)
    write_json(run / f"evaluation_{split}.json", result)
    return result
