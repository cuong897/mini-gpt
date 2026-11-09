"""One command, offline, CPU by default. Artifacts explicitly marked synthetic smoke."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", default="runs/smoke")
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise SystemExit("Chọn --root mới; không ghi đè run cũ")
    root.mkdir(parents=True)

    def run(*words):
        subprocess.run([sys.executable, "-m", "minigpt", *map(str, words)], check=True)

    data = root / "data"
    run("fixture", "--output", root / "fixture.jsonl")
    run("prepare", "--source", root / "fixture.jsonl", "--output", data, "--vocab-size", 512, "--kind", "synthetic")
    run("ngram", "--data", data, "--run", root / "ngram", "--context", 32, "--orders", 2, 3, "--alphas", 1, 10, "--kind", "smoke")
    for model in ["gpt", "lstm"]:
        run("train", "--config", f"configs/{model}_smoke.json", "--data", data, "--run", root / model, "--device", args.device)
    for model in ["ngram", "lstm", "gpt"]:
        run("evaluate", "--data", data, "--run", root / model, "--device", args.device)
        run("generate", "--data", data, "--run", root / model, "--device", args.device,
            "--prompt", "Buổi sáng, chúng tôi", "--top-p", .9, "--max-new-tokens", 30, "--output", root / model / "sample.json")
    subprocess.run([sys.executable, "scripts/summarize.py", "--runs", *[str(root / m) for m in ["ngram", "lstm", "gpt"]], "--output", str(root / "comparison")], check=True)


if __name__ == "__main__":
    main()
