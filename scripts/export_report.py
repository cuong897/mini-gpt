"""Export the pilot table directly from measured, compatible results."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.experiments import checked_result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/pilot")
    p.add_argument("--output", default="report/pilot_results.tex")
    args = p.parse_args()
    root = Path(args.root)
    checked = [checked_result(root / model) for model in ["ngram", "lstm", "gpt"]]
    results = [pair[0] for pair in checked]
    if any(row["kind"] != "pilot" or row["data_kind"] != "real" for row in results):
        raise ValueError("Chỉ nhận pilot trên dữ liệu thật")
    if len({(r["data_fingerprint"], r["tokens"], r["context"], r["protocol"]) for r in results}) != 1:
        raise ValueError("Các run không có giao thức/dữ liệu chung")
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lrr}", r"\toprule",
             r"Mô hình & Test NLL & Test PPL \\", r"\midrule"]
    for name, row in zip(["N-gram", "LSTM", "GPT"], results):
        lines.append(f"{name} & {row['loss']:.4f} & {row['perplexity']:.2f} " + r"\\")
    targets = results[0]["tokens"]
    budgets = ", ".join(f"{ev['model']}: {meta['tokens_seen']:,}" for ev, meta in checked if ev["model"] != "ngram")
    lines += [r"\bottomrule", r"\end{tabular}",
              "\\caption{Pilot kiểm chứng, chưa phải kết quả chính. Cùng test/tokenizer và " + f"{targets:,}" +
              " target. Ngân sách neural (target đã xử lý): " + budgets + ". N-gram đếm toàn bộ train.}",
              r"\label{tab:pilot}", r"\end{table}"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
