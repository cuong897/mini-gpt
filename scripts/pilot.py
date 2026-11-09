"""A short real-data integration run. Not evidence for final model quality."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/wiki_reviewed_v1")
    p.add_argument("--root", default="runs/reviewed_pilot")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    root = Path(args.root)
    if root.exists():
        raise SystemExit("Run đã tồn tại; chọn --root mới")
    root.mkdir(parents=True)
    def run(*words):
        subprocess.run([sys.executable, "-m", "minigpt", *map(str, words)], check=True)
    run("ngram", "--data", args.data, "--run", root / "ngram", "--context", 256,
        "--orders", 2, 3, "--alphas", 1, 10, "--kind", "pilot")
    for model in ["gpt", "lstm"]:
        run("train", "--config", f"configs/{model}_pilot.json", "--data", args.data,
            "--run", root / model, "--device", args.device)
    for model in ["ngram", "lstm", "gpt"]:
        run("evaluate", "--data", args.data, "--run", root / model, "--device", args.device)
        run("generate", "--data", args.data, "--run", root / model, "--device", args.device,
            "--prompt", "Việt Nam là một quốc gia", "--max-new-tokens", 40, "--top-p", .9,
            "--output", root / model / "sample.json")
    subprocess.run([sys.executable, "scripts/summarize.py", "--runs", *[str(root / m) for m in ["ngram", "lstm", "gpt"]],
                    "--output", str(root / "comparison")], check=True)


if __name__ == "__main__":
    main()
