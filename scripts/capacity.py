"""One synthetic optimizer step at official size; this is a hardware probe only."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from minigpt.common import environment, read_json, seed_all, write_json
from minigpt.engine import autocast_context, device_for
from minigpt.models import build_model, parameter_count, token_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gpt_main.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="artifacts/capacity.json")
    args = p.parse_args()
    config = read_json(args.config)
    seed_all(42)
    torch.set_num_threads(2)
    device = device_for(args.device)
    model = build_model(config, 8000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x = torch.randint(4, 8000, (config["batch_size"], config["architecture"]["context"]), device=device)
    y = torch.randint(4, 8000, x.shape, device=device)
    started = time.perf_counter()
    with autocast_context(device, config.get("precision", "fp32")):
        loss = token_loss(model(x).float(), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {"purpose": "synthetic hardware probe, NOT language-model experiment", "config": args.config,
              "parameters": parameter_count(model), "seconds_one_microbatch": time.perf_counter() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
              "loss_finite": bool(torch.isfinite(loss)), "environment": environment()}
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
