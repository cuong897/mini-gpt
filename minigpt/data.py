"""Tách tài liệu trước tokenizer. Không ghép hai tài liệu vào một context."""
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .common import BOS, EOS, PAD, SPECIALS, read_json, sha256, write_json


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def simhash(text):
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    shingles = Counter(" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2)))
    score = np.zeros(64, dtype=np.int64)
    for shingle, count in shingles.items():
        digest = hashlib.blake2b(shingle.encode(), digest_size=8).digest()
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int64)
        score += (bits * 2 - 1) * count
    return int.from_bytes(np.packbits(score >= 0).tobytes(), "big")


class Deduplicator:
    """Exact SHA256 + SimHash Hamming <= 3. Four bands guarantee candidate recall
    for that definition; SQLite keeps fingerprints off the Python heap.
    This does not detect every semantic duplicate or copied paragraph.
    """
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("CREATE TABLE docs (id INTEGER PRIMARY KEY, exact TEXT UNIQUE, sim TEXT);"
                              "CREATE TABLE bands (band INTEGER, value INTEGER, doc INTEGER);"
                              "CREATE INDEX band_idx ON bands(band,value);")

    def add(self, text):
        key = " ".join(text.casefold().split())
        exact = hashlib.sha256(key.encode()).hexdigest()
        if self.db.execute("SELECT 1 FROM docs WHERE exact=?", (exact,)).fetchone():
            return "exact"
        signature = simhash(key)
        bands = [(b, (signature >> (16 * b)) & 65535) for b in range(4)]
        candidates = set()
        for b, value in bands:
            candidates.update(r[0] for r in self.db.execute("SELECT doc FROM bands WHERE band=? AND value=?", (b, value)))
        for idx in candidates:
            other = int(self.db.execute("SELECT sim FROM docs WHERE id=?", (idx,)).fetchone()[0])
            if (signature ^ other).bit_count() <= 3:
                return "near"
        idx = self.db.execute("INSERT INTO docs(exact,sim) VALUES (?,?)", (exact, str(signature))).lastrowid
        self.db.executemany("INSERT INTO bands VALUES (?,?,?)", [(b, v, idx) for b, v in bands])
        return None

    def close(self):
        self.db.commit()
        self.db.close()


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class ParagraphGroups:
    """Keep connected documents with shared long paragraphs in one split.

    Paragraph keys are exact after casefold/whitespace normalization. This is
    intentionally a bounded overlap definition, not semantic deduplication.
    Only hashes and union-find indices stay in memory; article text is staged.
    """
    min_characters = 400

    def __init__(self, seed):
        self.seed = seed
        self.parent, self.sizes, self.keys, self.buckets = [], [], [], []
        self.owner = {}

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parent[right] = left
        self.sizes[left] += self.sizes[right]
        # Choose a representative independently of seed and input ordering.
        # Singletons retain the previous seeded document-hash assignment.
        if self.keys[right] < self.keys[left]:
            self.keys[left], self.buckets[left] = self.keys[right], self.buckets[right]

    def add(self, text, content_hash):
        index = len(self.parent)
        self.parent.append(index)
        self.sizes.append(1)
        self.keys.append(content_hash)
        self.buckets.append(hashlib.sha256((str(self.seed) + "\0" + text).encode()).hexdigest())
        for paragraph in text.split("\n\n"):
            canonical = " ".join(paragraph.casefold().split())
            if len(canonical) >= self.min_characters:
                key = hashlib.sha256(canonical.encode()).hexdigest()
                if key in self.owner:
                    self.union(index, self.owner[key])
                else:
                    self.owner[key] = index

    def metadata(self):
        roots = [i for i, parent in enumerate(self.parent) if i == parent]
        return {"method": "connected components of shared exact paragraphs after NFC/casefold/whitespace normalization",
                "min_paragraph_characters": self.min_characters,
                "groups": len(roots),
                "documents_in_multi_document_groups": sum(self.sizes[i] for i in roots if self.sizes[i] > 1),
                "largest_group_documents": max((self.sizes[i] for i in roots), default=0),
                "assignment": "seeded hash of text of member with smallest content SHA256; singleton assignment unchanged"}


def prepare(source, output, vocab_size=8000, seed=42, min_chars=200, ratios=(.9, .05, .05), kind="real"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Thư mục output phải trống để tránh trộn artifact cũ")
    if (vocab_size < 260 or min_chars < 0 or len(ratios) != 3
            or any(not math.isfinite(r) or r <= 0 for r in ratios)
            or abs(sum(ratios) - 1) > 1e-8):
        raise ValueError("Cần vocab >=260 và 3 tỷ lệ dương có tổng bằng 1")
    provenance_path = Path(str(source) + ".source.json")
    source_digest = sha256(source)
    provenance = read_json(provenance_path) if provenance_path.exists() else {"source": "local JSONL; see source sha256"}
    if "sha256" in provenance and provenance["sha256"] != source_digest:
        raise ValueError("SHA256 của corpus không khớp provenance nguồn")
    output.mkdir(parents=True, exist_ok=True)
    dedup = Deduplicator(output / "dedup.sqlite")
    groups = ParagraphGroups(seed)
    stats = Counter()
    staged = output / ".documents.staging.jsonl"
    try:
        with open(staged, "w", encoding="utf-8") as stream:
            for row in iter_jsonl(source):
                stats["input_documents"] += 1
                text = normalize(row["text"])
                if len(text) < min_chars:
                    stats["too_short"] += 1
                    continue
                duplicate = dedup.add(text)
                if duplicate:
                    stats[f"duplicate_{duplicate}"] += 1
                    continue
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                groups.add(text, content_hash)
                row.update(text=text, content_sha256=content_hash)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        dedup.close()
    # All overlap edges must be known before any split is assigned: a later
    # article can connect two earlier components transitively.
    files = {s: open(output / f"{s}.jsonl", "w", encoding="utf-8") for s in ["train", "validation", "test"]}
    try:
        for index, row in enumerate(iter_jsonl(staged)):
            root = groups.find(index)
            fraction = int(groups.buckets[root][:16], 16) / 2**64
            split = "train" if fraction < ratios[0] else "validation" if fraction < sum(ratios[:2]) else "test"
            row["split_group_sha256"] = groups.keys[root]
            files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[f"{split}_documents"] += 1
    finally:
        for f in files.values():
            f.close()
    staged.unlink()
    if any(stats[f"{s}_documents"] == 0 for s in files):
        raise ValueError("Tập rỗng sau chia; cần thêm tài liệu hoặc chọn tỷ lệ phù hợp")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator((r["text"] for r in iter_jsonl(output / "train.jsonl")), trainer=trainer)
    tokenizer.save(str(output / "tokenizer.json"))
    for i, symbol in enumerate(SPECIALS):
        assert tokenizer.token_to_id(symbol) == i
    split_stats = {}
    hashes = {"source": source_digest, "tokenizer.json": sha256(output / "tokenizer.json")}
    for split in files:
        offsets = [0]
        chars = unk = 0
        with open(output / f"{split}.bin", "wb") as f:
            for row in iter_jsonl(output / f"{split}.jsonl"):
                # User text cannot inject BOS/EOS via literal special-token strings.
                ids = encode_text(tokenizer, row["text"])
                seq = np.array([BOS] + ids + [EOS], dtype="<u4")
                f.write(seq.tobytes())
                offsets.append(offsets[-1] + len(seq))
                chars += len(row["text"])
                unk += ids.count(3)
        np.save(output / f"{split}.offsets.npy", np.array(offsets, dtype=np.int64))
        split_stats[split] = {"documents": len(offsets) - 1, "stored_tokens": offsets[-1],
                              "target_tokens": offsets[-1] - len(offsets) + 1, "characters": chars, "unk_tokens": unk}
        for suffix in ["jsonl", "bin", "offsets.npy"]:
            name = f"{split}.{suffix}"
            hashes[name] = sha256(output / name)
    total = sum(v["documents"] for v in split_stats.values())
    for v in split_stats.values():
        v["document_ratio"] = v["documents"] / total
        v["token_ratio"] = v["target_tokens"] / sum(x["target_tokens"] for x in split_stats.values())
    manifest = {"kind": kind, "seed": seed, "requested_ratios": ratios, "min_chars": min_chars,
                "provenance": provenance,
                "dedup": "casefold-whitespace SHA256; word-trigram 64bit SimHash Hamming<=3; retain first",
                "split_grouping": groups.metadata(),
                "cleaning": "NFC; preserve accents/case/newlines; remove control chars; normalize spaces",
                "vocab_size": tokenizer.get_vocab_size(), "special_ids": dict(zip(SPECIALS, range(4))),
                "counts": dict(stats), "splits": split_stats, "hashes": hashes}
    write_json(output / "manifest.json", manifest)
    return manifest


def encode_text(tokenizer, text):
    # Encode ordinary text with special recognition disabled, while sharing trained BPE.
    if not any(symbol in text for symbol in SPECIALS):
        return tokenizer.encode(normalize(text), add_special_tokens=False).ids
    plain = Tokenizer.from_str(tokenizer.to_str())
    for token in plain.get_added_tokens_decoder().values():
        if token.special and token.content in text:
            # Byte-level BPE alone handles arbitrary strings, including literal <eos>.
            raw = json.loads(plain.to_str())
            raw["added_tokens"] = []
            plain = Tokenizer.from_str(json.dumps(raw))
            break
    return plain.encode(normalize(text), add_special_tokens=False).ids


class Documents:
    def __init__(self, root, split):
        root = Path(root)
        self.tokens = np.memmap(root / f"{split}.bin", dtype="<u4", mode="r")
        self.offsets = np.load(root / f"{split}.offsets.npy", mmap_mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]


class BlockDataset(Dataset):
    """Targets cover tokens 1..len(doc)-1 exactly once, including EOS.
    A later block has real previous token as its first input, no artificial BOS.
    Hidden state resets per block for both neural architectures.
    """
    def __init__(self, root, split, context):
        if context < 1:
            raise ValueError("context phải >=1")
        self.docs, self.context = Documents(root, split), context
        counts = (np.diff(self.docs.offsets) - 2) // context + 1
        self.cumulative = np.cumsum(counts)

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        doc_id = int(np.searchsorted(self.cumulative, index, side="right"))
        first = 0 if doc_id == 0 else int(self.cumulative[doc_id - 1])
        start = (index - first) * self.context
        seq = self.docs[doc_id][start:start + self.context + 1].astype(np.int64)
        x, y = torch.full((self.context,), PAD), torch.full((self.context,), PAD)
        x[:len(seq) - 1] = torch.from_numpy(seq[:-1].copy())
        y[:len(seq) - 1] = torch.from_numpy(seq[1:].copy())
        return x, y


def verify_artifacts(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    for name, digest in manifest["hashes"].items():
        if name != "source" and sha256(root / name) != digest:
            raise ValueError(f"Artifact bị thay đổi: {name}")
    return sha256(root / "manifest.json")
