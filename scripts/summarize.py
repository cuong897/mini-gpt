"""Generate tables and plots exclusively from existing measured artifacts."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.experiments import checked_result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.runs:
        root = Path(name)
        ev, meta = checked_result(root)
        rows.append({"run": str(root), "model": ev["model"], "kind": ev["kind"], "data_kind": ev["data_kind"],
                     "parameters": meta.get("parameters", "N/A (counts)"), "count_entries": meta.get("count_entries", ""),
                     "context": ev["context"], "data_fingerprint": ev["data_fingerprint"], "protocol": ev["protocol"],
                     "tokens_evaluated": ev["tokens"], "test_loss": ev["loss"], "test_perplexity": ev["perplexity"],
                     "training_tokens": meta.get("tokens_seen", meta.get("train_unique_target_tokens")),
                     "best_validation_loss": meta.get("best_validation_loss", meta.get("validation", {}).get("loss")),
                     "checkpoint": ev["checkpoint"], "checkpoint_step": ev.get("checkpoint_step", "legacy/unrecorded"),
                     "checkpoint_tokens_seen": ev.get("checkpoint_tokens_seen", "legacy/unrecorded"),
                     "checkpoint_sha256": ev.get("checkpoint_sha256") or "legacy/unrecorded",
                     "checkpoint_metadata_verified": meta.get("checkpoint_metadata_verified", False),
                     "checkpoint_binding_verified": meta.get("checkpoint_binding_verified", False),
                     "order": meta.get("order", ""), "alpha": meta.get("alpha", ""),
                     "seconds": meta.get("elapsed_train_seconds", meta.get("fit_and_tune_seconds")),
                     "budget_completed": meta.get("budget_completed", "N/A")})
    signatures = {(r["data_fingerprint"], r["context"], r["protocol"], r["tokens_evaluated"], r["kind"], r["data_kind"]) for r in rows}
    if len(signatures) != 1:
        raise ValueError("Không gộp bảng perplexity khác tokenizer/dữ liệu/context/protocol/kind")
    scaling = [r for r in rows if r["model"] == "gpt"]
    if len(scaling) >= 3:
        if (len({r["training_tokens"] for r in scaling}) != 1
                or not all(r["budget_completed"] is True for r in scaling)
                or not all(r["checkpoint_binding_verified"] for r in scaling)
                or len({r["parameters"] for r in scaling}) < 3):
            raise ValueError("Scaling cần ba kích thước khác nhau và cùng ngân sách token hoàn thành")
    with open(output / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Kết quả thực chạy — {rows[0]['kind']} / {rows[0]['data_kind']}", "",
             ("Smoke/pilot chỉ xác minh chương trình; không đại diện chất lượng mô hình chính."
              if rows[0]["kind"] in ["smoke", "pilot"] else "Kết quả từ các run chính; xem cấu hình, ngân sách và giới hạn giao thức."), "",
             "| Run | Tham số | Val loss | Test loss | Test PPL | Target tokens | Training tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        ppl = f"{row['test_perplexity']:.3f}" if row["test_perplexity"] is not None else "overflow"
        lines.append(f"| {row['run']} | {row['parameters']} | {row['best_validation_loss']:.4f} | {row['test_loss']:.4f} | {ppl} | {row['tokens_evaluated']} | {row['training_tokens']} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in args.runs:
        path = Path(name) / "metrics.jsonl"
        if path.exists():
            history = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            axes[0].plot([r["tokens_seen"] for r in history], [r["train_loss"] for r in history], label=Path(name).name)
            val = [r for r in history if "validation_loss" in r]
            axes[0].plot([r["tokens_seen"] for r in val], [r["validation_loss"] for r in val], "o--")
    axes[0].set(xlabel="Training tokens (including repetitions)", ylabel="NLL per token")
    if axes[0].lines:
        axes[0].legend()
    axes[1].bar([Path(r["run"]).name for r in rows], [r["test_loss"] for r in rows])
    axes[1].set(ylabel="Test NLL per token")
    fig.suptitle(f"Measured {rows[0]['kind']} / {rows[0]['data_kind']}")
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=180)
    plt.close(fig)
    if len(scaling) >= 3:
        fig, ax = plt.subplots(figsize=(5, 4))
        scaling.sort(key=lambda r: r["parameters"])
        ax.plot([r["parameters"] for r in scaling], [r["test_loss"] for r in scaling], "o-")
        ax.set(xlabel="Parameters", ylabel="Test NLL per token", title="Measured size comparison (not a scaling law)")
        fig.tight_layout()
        fig.savefig(output / "scaling.png", dpi=180)
        plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
