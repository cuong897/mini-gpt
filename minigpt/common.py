import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PROTOCOL = "document-block-v1: each target once; BOS excluded; EOS included; reset each block"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    return {"python": random.getstate(), "numpy": list(np.random.get_state()[:1]) +
            [np.random.get_state()[1].tolist()] + list(np.random.get_state()[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def environment():
    import importlib.metadata
    packages = {}
    for name in ["torch", "numpy", "tokenizers", "datasets", "tensorboard", "pytest", "matplotlib", "psutil"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(Path.cwd())
    return {"recorded_utc": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
            "python": sys.version, "packages": packages, "cuda_build": torch.version.cuda,
            "gpus": [{"name": torch.cuda.get_device_name(i), "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())], "free_disk_bytes": disk.free}


def metrics(nll_sum, tokens):
    if tokens <= 0:
        raise ValueError("Không có target token để đánh giá")
    loss = nll_sum / tokens
    return {"nll_sum": nll_sum, "tokens": tokens, "loss": loss,
            "perplexity": math.exp(loss) if loss < 709 else None}


def atomic_save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
