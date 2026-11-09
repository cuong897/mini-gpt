import argparse
import json
from pathlib import Path

from .common import environment, write_json


def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Mini-GPT tiếng Việt, huấn luyện từ đầu")
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env")
    env.add_argument("--output", default="artifacts/environment.json")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--output", default="data/raw/wiki.jsonl")
    fetch.add_argument("--documents", type=int, default=10000)
    fetch.add_argument("--revision", default="main")
    fetch.add_argument("--seed", type=int, default=42)
    fetch.add_argument("--buffer-size", type=int, default=1000)
    fetch.add_argument("--min-chars", type=int, default=1000)
    fetch.add_argument("--max-chars", type=int, default=100000)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--output", default="data/raw/fixture.jsonl")
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--vocab-size", type=int, default=8000)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--min-chars", type=int, default=200)
    prep.add_argument("--ratios", type=float, nargs=3, default=(.9, .05, .05))
    prep.add_argument("--kind", choices=["real", "synthetic"], default="real")
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--run", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume")
    train.add_argument("--continue-from", help="last.pt cho giai đoạn học mới; giữ optimizer/RNG, đặt lại lịch và bộ đếm")
    train.add_argument("--stop-after", type=int)
    ng = sub.add_parser("ngram")
    ng.add_argument("--data", required=True)
    ng.add_argument("--run", required=True)
    ng.add_argument("--context", type=int, default=256)
    ng.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    ng.add_argument("--alphas", type=float, nargs="+", default=[1., 10., 100.])
    ng.add_argument("--kind", choices=["smoke", "pilot", "main"], default="main")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", required=True)
    ev.add_argument("--run", required=True)
    ev.add_argument("--split", choices=["validation", "test"], default="test")
    ev.add_argument("--device", default="auto")
    ev.add_argument("--checkpoint", default="best.pt")
    ev.add_argument("--batch-size", type=int, default=8)
    gen = sub.add_parser("generate")
    gen.add_argument("--data", required=True)
    gen.add_argument("--run", required=True)
    gen.add_argument("--prompt", default="Hôm nay, tôi")
    gen.add_argument("--max-new-tokens", type=int, default=100)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--temperature", type=float, default=1.)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=1.)
    gen.add_argument("--greedy", action="store_true")
    gen.add_argument("--device", default="auto")
    gen.add_argument("--checkpoint", default="best.pt")
    gen.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "env":
        result = environment()
        write_json(args["output"], result)
    elif command == "fetch":
        result = fetch_wiki(**args)
    elif command == "fixture":
        result = make_fixture(args["output"])
    elif command == "prepare":
        from .data import prepare
        result = prepare(**args)
    elif command == "train":
        from .engine import train
        args["config_path"] = args.pop("config")
        args["device_name"] = args.pop("device")
        result = train(**args)
    elif command == "ngram":
        from .ngram import train_ngram
        result = train_ngram(**args)
    elif command == "evaluate":
        from .engine import evaluate_run
        args["device_name"] = args.pop("device")
        result = evaluate_run(**args)
    else:
        from .generation import generate
        output = args.pop("output")
        args["device_name"] = args.pop("device")
        result = generate(**args)
        write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def fetch_wiki(output, documents, revision="main", seed=42, buffer_size=1000, min_chars=1000, max_chars=100000):
    import os
    from itertools import islice
    from .common import sha256
    from .data import normalize
    if documents < 1 or buffer_size < 1 or min_chars < 0 or max_chars < min_chars:
        raise ValueError("documents/buffer_size >=1 và 0<=min_chars<=max_chars")
    output = Path(output)
    if output.exists():
        raise ValueError("Không ghi đè corpus đã tải")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset_id, subset = "wikimedia/wikipedia", "20231101.vi"
    commit = HfApi().dataset_info(dataset_id, revision=revision).sha
    stream = load_dataset(dataset_id, subset, split="train", revision=commit, streaming=True)
    # Remove tiny taxonomic stubs before local shuffling, to keep the subset useful.
    stream = stream.filter(lambda row: min_chars <= len(normalize(row["text"])) <= max_chars)
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)
    tmp = output.with_suffix(".partial")
    count = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in islice(stream, documents):
            row["source"] = dataset_id
            row["source_revision"] = commit
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if count != documents:
        raise ValueError(f"Nguồn chỉ cung cấp {count}/{documents} tài liệu; giữ .partial để kiểm tra")
    tmp.replace(output)
    metadata = {"dataset": dataset_id, "subset": subset, "revision": commit, "documents": count,
                "shuffle_seed": seed, "shuffle_buffer": buffer_size, "min_chars": min_chars, "max_chars": max_chars, "sha256": sha256(output),
                "license_note": "Dataset card: CC BY-SA 3.0 / GFDL; see source terms for exceptions. Keep article URLs and attribution.",
                "dataset_card": "https://huggingface.co/datasets/wikimedia/wikipedia"}
    write_json(str(output) + ".source.json", metadata)
    return metadata


def make_fixture(output):
    """Authored artificial fixture, intentionally varied; never research evidence."""
    import random
    output = Path(output)
    if output.exists():
        raise ValueError("Fixture đã tồn tại")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(731)
    subjects = ["mùa xuân", "khu vườn", "ngôi trường", "con sông", "gia đình", "thư viện", "bầu trời", "thành phố"]
    actions = ["đọc sách", "trồng cây", "học tiếng Việt", "nghe chim hót", "kể chuyện", "tìm hiểu khoa học", "đi dạo", "viết bài"]
    times = ["Buổi sáng", "Chiều hôm qua", "Vào ngày nghỉ", "Khi trời mưa", "Mỗi mùa hè", "Sau giờ học"]
    with open(output, "w", encoding="utf-8") as f:
        for i in range(120):
            sentences = [f"{rng.choice(times)}, chúng tôi cùng {rng.choice(actions)} và nói về {rng.choice(subjects)}." for _ in range(5)]
            row = {"id": f"synthetic-{i}", "text": " ".join(sentences), "source": "authored synthetic fixture; no experimental claims"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"documents": 120, "kind": "synthetic", "output": str(output)}


if __name__ == "__main__":
    main()
