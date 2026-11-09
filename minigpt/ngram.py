"""Sparse n-gram counts on disk; recursive Dirichlet interpolation/backoff."""
import math
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import BOS, PAD, metrics, read_json, write_json
from .data import BlockDataset


def context_key(history):
    return np.asarray(history, dtype="<u4").tobytes()


class NGram:
    def __init__(self, path, vocab_size, order=3, alpha=10., delta=.1, readonly=True):
        if (not isinstance(order, int) or isinstance(order, bool) or order < 1
                or vocab_size < 4 or not math.isfinite(alpha) or alpha <= 0
                or not math.isfinite(delta) or delta <= 0):
            raise ValueError("order nguyên>=1, vocab>=4; alpha,delta hữu hạn>0")
        self.path, self.vocab_size, self.order = str(path), vocab_size, order
        self.alpha, self.delta = alpha, delta
        if readonly:
            self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.db = sqlite3.connect(path)
            self.db.executescript("CREATE TABLE counts(ctx BLOB, token INTEGER, count INTEGER, PRIMARY KEY(ctx,token)) WITHOUT ROWID;"
                                  "CREATE TABLE totals(ctx BLOB PRIMARY KEY, total INTEGER) WITHOUT ROWID;")

    def fit(self, dataset, flush_entries=100000):
        counts = Counter()

        def flush():
            totals = Counter()
            for (ctx, _), count in counts.items():
                totals[ctx] += count
            with self.db:
                self.db.executemany("INSERT INTO counts VALUES (?,?,?) ON CONFLICT(ctx,token) DO UPDATE SET count=count+excluded.count",
                                    ((ctx, token, count) for (ctx, token), count in counts.items()))
                self.db.executemany("INSERT INTO totals VALUES (?,?) ON CONFLICT(ctx) DO UPDATE SET total=total+excluded.total", totals.items())
            counts.clear()

        for x, y in dataset:
            history = []
            for previous, target in zip(x.tolist(), y.tolist()):
                if target == PAD:
                    break
                history.append(previous)
                for length in range(min(self.order - 1, len(history)) + 1):
                    ctx = context_key(history[-length:] if length else [])
                    counts[(ctx, target)] += 1
                if len(counts) >= flush_entries:
                    flush()
        if counts:
            flush()
        self.total.cache_clear()
        self.count.cache_clear()

    @lru_cache(maxsize=50000)
    def total(self, ctx):
        row = self.db.execute("SELECT total FROM totals WHERE ctx=?", (ctx,)).fetchone()
        return row[0] if row else 0

    @lru_cache(maxsize=100000)
    def count(self, ctx, token):
        row = self.db.execute("SELECT count FROM counts WHERE ctx=? AND token=?", (ctx, token)).fetchone()
        return row[0] if row else 0

    def probability(self, token, history):
        if token in (PAD, BOS):
            return 0.
        p = (self.count(b"", token) + self.delta) / (self.total(b"") + self.delta * (self.vocab_size - 2))
        for length in range(1, min(self.order - 1, len(history)) + 1):
            ctx = context_key(history[-length:])
            p = (self.count(ctx, token) + self.alpha * p) / (self.total(ctx) + self.alpha)
        return p

    def distribution(self, history):
        p = np.full(self.vocab_size, self.delta, dtype=np.float64)
        p[[PAD, BOS]] = 0
        for token, count in self.db.execute("SELECT token,count FROM counts WHERE ctx=?", (b"",)):
            p[token] += count
        p /= self.total(b"") + self.delta * (self.vocab_size - 2)
        for length in range(1, min(self.order - 1, len(history)) + 1):
            ctx = context_key(history[-length:])
            p *= self.alpha
            for token, count in self.db.execute("SELECT token,count FROM counts WHERE ctx=?", (ctx,)):
                p[token] += count
            p /= self.total(ctx) + self.alpha
        return p

    def evaluate(self, dataset):
        nll, tokens = 0., 0
        for x, y in dataset:
            history = []
            for previous, target in zip(x.tolist(), y.tolist()):
                if target == PAD:
                    break
                history.append(previous)
                nll -= math.log(self.probability(target, history))
                tokens += 1
        return metrics(nll, tokens)

    def close(self):
        self.total.cache_clear()
        self.count.cache_clear()
        self.db.close()


def train_ngram(data, run, context=256, orders=(2, 3, 4), alphas=(1., 10., 100.), delta=.1, kind="main"):
    from .common import PROTOCOL, environment, sha256
    from .data import verify_artifacts
    import time
    if (not orders or not alphas or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in orders)
            or any(not math.isfinite(a) or a <= 0 for a in alphas)
            or not math.isfinite(delta) or delta <= 0 or context < 1):
        raise ValueError("Cần orders nguyên>=1, alphas/delta hữu hạn>0, context>=1")
    fingerprint = verify_artifacts(data)
    manifest = read_json(Path(data) / "manifest.json")
    if kind == "main" and manifest["kind"] == "synthetic":
        raise ValueError("Không huấn luyện chính trên dữ liệu giả")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "ngram.sqlite").exists():
        raise ValueError("Run n-gram đã tồn tại; chọn thư mục mới")
    started = time.perf_counter()
    model = NGram(run / "ngram.sqlite", manifest["vocab_size"], max(orders), alphas[0], delta, readonly=False)
    train = BlockDataset(data, "train", context)
    model.fit(train)
    trials = []
    for order in orders:
        for alpha in alphas:
            model.order, model.alpha = order, alpha
            result = model.evaluate(BlockDataset(data, "validation", context))
            trials.append({"order": order, "alpha": alpha, **result})
            print(json_line(trials[-1]), flush=True)
    best = min(trials, key=lambda r: r["loss"])
    entries = model.db.execute("SELECT count(*) FROM counts").fetchone()[0]
    model.close()
    metadata = {"model": "ngram", "kind": kind, "data_kind": manifest["kind"], "data_fingerprint": fingerprint,
                "vocab_size": manifest["vocab_size"], "context": context, "order": best["order"], "alpha": best["alpha"],
                "delta": delta, "validation": best, "trials": trials, "count_entries": entries,
                "fit_and_tune_seconds": time.perf_counter() - started, "train_unique_target_tokens": manifest["splits"]["train"]["target_tokens"],
                "environment": environment(), "protocol": PROTOCOL, "counts_sha256": sha256(run / "ngram.sqlite")}
    write_json(run / "ngram.json", metadata)
    return metadata


def json_line(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)
