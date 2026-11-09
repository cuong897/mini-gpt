import hashlib
import json
import unicodedata

import numpy as np
import pytest
from tokenizers import Tokenizer

from minigpt.__main__ import fetch_wiki, make_fixture
from minigpt.common import BOS, EOS, PAD, read_json, sha256, write_json
from minigpt.data import (BlockDataset, Deduplicator, Documents, encode_text, iter_jsonl,
                          normalize, prepare, simhash, verify_artifacts)


@pytest.fixture
def prepared(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "data"
    make_fixture(source)
    prepare(source, output, vocab_size=320, kind="synthetic", ratios=(.7, .15, .15))
    return output


def test_unicode_and_special_literals(prepared):
    text = "Tiếng Việt rất đẹp.\n\nMột ngày mới."
    assert normalize(unicodedata.normalize("NFD", text)) == text
    tok = Tokenizer.from_file(str(prepared / "tokenizer.json"))
    assert tok.decode(encode_text(tok, text)) == text
    literal = "Tôi viết <eos> và <bos> trong sách."
    ids = encode_text(tok, literal)
    assert not set(ids) & {PAD, BOS, EOS, 3}
    assert tok.decode(ids) == literal


def test_dedup_exact_and_near(tmp_path):
    dedup = Deduplicator(tmp_path / "dedup.sqlite")
    text = ("Mỗi sáng chúng tôi cùng học tiếng Việt trong khu vườn yên tĩnh. " * 100).strip()
    assert dedup.add(text) is None
    # Deduplicator consumes normalized text; pipeline normalizes before add().
    assert dedup.add(normalize(unicodedata.normalize("NFD", text))) == "exact"
    modified = text + " Một chiếc lá rơi."
    assert (simhash(text) ^ simhash(modified)).bit_count() <= 3
    assert dedup.add(modified) == "near"
    dedup.close()


def test_boundaries_no_overlap_and_fingerprints(prepared):
    all_hashes = []
    signatures = []
    for split in ["train", "validation", "test"]:
        rows = list(iter_jsonl(prepared / f"{split}.jsonl"))
        all_hashes.extend(row["content_sha256"] for row in rows)
        signatures.extend((split, simhash(row["text"])) for row in rows)
        docs = Documents(prepared, split)
        for context in [1, 7, 32, 1000]:
            blocks = BlockDataset(prepared, split, context)
            targets = []
            for x, y in blocks:
                mask = y.ne(PAD)
                assert mask.any()
                assert not y[mask].eq(BOS).any()
                targets.extend(y[mask].tolist())
            expected = [int(t) for doc in docs for t in doc[1:]]
            assert targets == expected
            assert targets.count(EOS) == len(docs)
    assert len(all_hashes) == len(set(all_hashes))
    for i, (left_split, left) in enumerate(signatures):
        for right_split, right in signatures[i + 1:]:
            if left_split != right_split:
                assert (left ^ right).bit_count() > 3
    verify_artifacts(prepared)
    with open(prepared / "test.bin", "ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Artifact"):
        verify_artifacts(prepared)


def test_tokenizer_train_only_when_heldout_changes(tmp_path):
    source = tmp_path / "original.jsonl"
    make_fixture(source)
    first = tmp_path / "first"
    prepare(source, first, vocab_size=512, ratios=(.7, .15, .15), kind="synthetic")
    train_texts = [r["text"] for r in iter_jsonl(first / "train.jsonl")]
    # Replace only held-out content; search a hash bucket without moving train.
    train_hashes = set(train_texts)
    rows = list(iter_jsonl(source))
    for row in rows:
        if row["text"] in train_hashes:
            continue
        base = " ".join("Zyxw" + hashlib.sha256(f"{row['id']}:{j}".encode()).hexdigest()[:12] for j in range(30))
        original_digest = hashlib.sha256(("42\0" + normalize(row["text"])).encode()).hexdigest()
        was_test = int(original_digest[:16], 16) / 2**64 >= .85
        for nonce in range(10000):
            candidate = f"{nonce}: {base} {row['id']}"
            digest = hashlib.sha256(("42\0" + normalize(candidate)).encode()).hexdigest()
            fraction = int(digest[:16], 16) / 2**64
            if (fraction >= .85) if was_test else (.7 <= fraction < .85):
                row["text"] = candidate
                break
    altered = tmp_path / "altered.jsonl"
    altered.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    second = tmp_path / "second"
    prepare(altered, second, vocab_size=512, ratios=(.7, .15, .15), kind="synthetic")
    assert [r["text"] for r in iter_jsonl(second / "train.jsonl")] == train_texts
    assert read_json(first / "tokenizer.json")["model"] == read_json(second / "tokenizer.json")["model"]


def test_reject_invalid_arguments_and_false_provenance_before_writing(tmp_path):
    source = tmp_path / "fixture.jsonl"
    make_fixture(source)
    for ratios in [(float("nan"), .1, .1), (.9, float("inf"), .1), (.9, 0, .1)]:
        with pytest.raises(ValueError):
            prepare(source, tmp_path / "bad", ratios=ratios)
        assert not (tmp_path / "bad").exists()
    write_json(str(source) + ".source.json", {"sha256": "0" * 64, "revision": "incorrect"})
    with pytest.raises(ValueError, match="provenance"):
        prepare(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="min_chars"):
        fetch_wiki(tmp_path / "fetch.jsonl", documents=1, min_chars=1000, max_chars=20)
