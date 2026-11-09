"""Download official ACL style at a resolved commit; preserve hashes and provenance."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "mini-gpt-student-report"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="master")
    p.add_argument("--output", default="report")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    commit = json.loads(get(f"https://api.github.com/repos/acl-org/acl-style-files/commits/{args.revision}"))["sha"]
    files = {}
    for name in ["acl.sty", "acl_natbib.bst"]:
        path = root / name
        if path.exists():
            raise ValueError(f"Không ghi đè {path}; kiểm tra provenance hoặc chọn --output khác")
        content = get(f"https://raw.githubusercontent.com/acl-org/acl-style-files/{commit}/{name}")
        path.write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    metadata = {"repository": "https://github.com/acl-org/acl-style-files", "revision": commit, "sha256": files}
    (root / "acl_upstream.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
