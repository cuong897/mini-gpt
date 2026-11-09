"""Copy small measured artifacts; never copy corpus or checkpoint weights."""
import argparse
import json
import shutil
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.experiments import checked_result
from minigpt.data import verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/reviewed_pilot")
    parser.add_argument("--data", default="data/wiki_reviewed_v1")
    parser.add_argument("--source-metadata", default="data/raw/wiki_filtered.jsonl.source.json")
    parser.add_argument("--output", default="artifacts/reviewed_pilot")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Thư mục evidence phải trống; không trộn log/run cũ")
    fingerprint = hashlib.sha256((Path(args.data) / "manifest.json").read_bytes()).hexdigest()
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    source_metadata = json.loads(Path(args.source_metadata).read_text(encoding="utf-8"))
    if source_metadata.get("sha256") != manifest["hashes"]["source"]:
        raise ValueError("Source metadata không khớp SHA256 nguồn của manifest")
    for key, value in manifest.get("provenance", {}).items():
        if source_metadata.get(key) != value:
            raise ValueError(f"Source metadata không khớp provenance {key}")
    verify_artifacts(args.data)
    for model in ["ngram", "lstm", "gpt"]:
        result, _ = checked_result(Path(args.root) / model)
        if result["data_fingerprint"] != fingerprint:
            raise ValueError(f"Evidence không cùng dữ liệu: {model}")
        sample_path = Path(args.root) / model / "sample.json"
        if sample_path.exists():
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            if (sample.get("data_fingerprint") != fingerprint
                    or sample.get("checkpoint_sha256") != result.get("checkpoint_sha256")):
                raise ValueError(f"Sample không cùng dữ liệu/checkpoint với evaluation: {model}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.data) / "manifest.json", output / "data_manifest.json")
    shutil.copy2(args.source_metadata, output / "data_source.json")
    for model in ["ngram", "lstm", "gpt"]:
        target = output / model
        target.mkdir(exist_ok=True)
        for name in ["config.json", "environment.json", "summary.json", "ngram.json", "metrics.jsonl", "evaluation_test.json", "sample.json"]:
            source = Path(args.root) / model / name
            if source.exists():
                shutil.copy2(source, target / name)
    print(output.resolve())


if __name__ == "__main__":
    main()
