"""Regression for copied paragraphs across otherwise different Wikipedia articles."""
import hashlib
import json
import random

from minigpt.data import iter_jsonl, normalize, prepare
from minigpt.common import read_json


def test_long_shared_paragraph_components_never_cross_splits(tmp_path):
    # Make three genuinely different articles that the previous document hash
    # split sends to train/validation/test. A--B and B--C share long paragraphs;
    # A and C do not, so pairwise reassignment is insufficient.
    rng = random.Random(6001)
    def words(count):
        return " ".join(f"{rng.getrandbits(64):016x}" for _ in range(count))

    first, second = words(32), words(32)
    texts = [words(180) + "\n\n" + first,
             words(180) + "\n\n" + first.upper().replace(" ", "  \t") + "\n\n" + second,
             words(180) + "\n\n" + second]
    buckets = [(0, .7), (.7, .85), (.85, 1)]
    special = []
    for index, (text, (lo, hi)) in enumerate(zip(texts, buckets)):
        for nonce in range(10000):
            candidate = f"{nonce}: {text}"
            digest = hashlib.sha256(("42\0" + normalize(candidate)).encode()).hexdigest()
            if lo <= int(digest[:16], 16) / 2**64 < hi:
                special.append({"id": f"copied-{index}", "text": candidate})
                break
        else:
            raise AssertionError("Could not construct regression articles")

    source = tmp_path / "original.jsonl"
    # Distinct filler articles keep all three splits nonempty without the
    # retain-first document deduplicator changing which fixtures survive.
    rows = [{"id": f"filler-{i}", "text": words(80)} for i in range(60)] + special
    assignments = []
    for index, ordered in enumerate((rows, list(reversed(rows)))):
        source.write_text("".join(json.dumps(r) + "\n" for r in ordered), encoding="utf-8")
        output = tmp_path / f"prepared-{index}"
        prepare(source, output, vocab_size=320, ratios=(.7, .15, .15), kind="synthetic")
        assigned = {r['id']: (split, r.get('split_group_sha256'))
                    for split in ('train', 'validation', 'test')
                    for r in iter_jsonl(output / f'{split}.jsonl')}
        assert all(r['id'] in assigned for r in special), "Test must exercise grouping, not doc dedup"
        assert len({assigned[r['id']][0] for r in special}) == 1
        assert len({assigned[r['id']][1] for r in special}) == 1
        assert assigned[special[0]['id']][1] is not None
        grouping = read_json(output / 'manifest.json')['split_grouping']
        assert grouping['min_paragraph_characters'] == 400
        assert grouping['largest_group_documents'] == 3
        assert grouping['documents_in_multi_document_groups'] == 3
        assignments.append(assigned)
    assert assignments[0] == assignments[1], "Groups and split assignments must be order-independent"
