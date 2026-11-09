"""Evaluate a fixed prompt set without hand-picking outputs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import read_json, write_json
from minigpt.generation import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prompts", default="configs/prompts.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    settings = [{"greedy": True}, {"temperature": .7}, {"temperature": 1.}, {"temperature": 1.3},
                {"temperature": 1., "top_k": 40}, {"temperature": 1., "top_p": .9}]
    rows = []
    checkpoint_hash = None
    for i, prompt in enumerate(read_json(args.prompts)):
        for setting in settings:
            for seed in args.seeds:
                row = generate(args.data, args.run, prompt, seed=seed, max_new_tokens=args.max_new_tokens,
                               device_name=args.device, **setting)
                if checkpoint_hash is None:
                    checkpoint_hash = row["checkpoint_sha256"]
                elif row["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError("Checkpoint thay đổi giữa các mẫu sweep; không trộn các model vào cùng thí nghiệm")
                row["prompt_id"] = i
                tokens = row["generated_ids"]
                row["distinct_1"] = len(set(tokens)) / len(tokens) if tokens else None
                pairs = list(zip(tokens, tokens[1:]))
                row["distinct_2"] = len(set(pairs)) / len(pairs) if pairs else None
                rows.append(row)
                print(f"prompt {i}, seed {seed}, {setting}", flush=True)
    write_json(args.output, rows)


if __name__ == "__main__":
    main()
