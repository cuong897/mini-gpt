"""Interactive terminal demo; no server or additional dependencies."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import write_json
from minigpt.generation import generate


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print("Mini-GPT tiếng Việt. Để trống prompt để thoát.")
    i = 0
    while True:
        prompt = input("Prompt: ").strip()
        if not prompt:
            break
        try:
            temperature = float(input("Temperature [0.8]: ") or .8)
            top_k = int(input("Top-k, 0 để tắt [40]: ") or 40)
            top_p = float(input("Top-p [0.9]: ") or .9)
            seed = int(input("Seed [42]: ") or 42)
            result = generate(args.data, args.run, prompt, seed=seed, temperature=temperature,
                              top_k=top_k, top_p=top_p, device_name=args.device)
        except ValueError as error:
            print(error)
            continue
        print(prompt + result["continuation"])
        write_json(Path(args.run) / "demo" / f"sample-{i:04d}.json", result)
        i += 1


if __name__ == "__main__":
    main()
