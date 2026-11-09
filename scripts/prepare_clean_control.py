"""Change train prose only; freeze tokenizer and heldout artifacts for a paired experiment."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.cleaning import clean_prose
from minigpt.common import BOS, EOS, sha256, read_json, write_json
from minigpt.data import Deduplicator, encode_text, iter_jsonl, verify_artifacts


def paragraph_keys(text):
    return {' '.join(p.casefold().split()) for p in text.split('\n\n') if len(' '.join(p.casefold().split())) >= 400}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    source, output, evidence = map(Path, [args.source, args.output, args.evidence])
    parent_fingerprint = verify_artifacts(source)
    if output.exists():
        raise ValueError('Choose a new output directory')
    output.mkdir(parents=True)
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(read_json(source/'manifest.json'))
    heldout_keys = set()
    dedup = Deduplicator(output/'dedup.sqlite')
    for split in ['validation', 'test']:
        for row in iter_jsonl(source/f'{split}.jsonl'):
            heldout_keys.update(paragraph_keys(row['text']))
            dedup.add(row['text'])
        for suffix in ['jsonl', 'bin', 'offsets.npy']:
            shutil.copyfile(source/f'{split}.{suffix}', output/f'{split}.{suffix}')
    shutil.copyfile(source/'tokenizer.json', output/'tokenizer.json')
    tokenizer = Tokenizer.from_file(str(output/'tokenizer.json'))
    stats, seen, retained = Counter(), set(), []
    offsets, characters = [0], 0
    audit = []
    with open(output/'train.jsonl', 'w', encoding='utf-8') as stream, open(output/'train.bin', 'wb') as binary:
        for row in iter_jsonl(source/'train.jsonl'):
            stats['input_documents'] += 1
            before = row['text']
            text, counts = clean_prose(before)
            stats.update(counts)
            reason = None
            if len(text) < 600 or len(text) < .2*len(before):
                reason = 'too_little_prose'
            elif paragraph_keys(text) & heldout_keys:
                reason = 'heldout_paragraph_overlap'
            canonical = ' '.join(text.casefold().split())
            if canonical in seen:
                reason = 'duplicate_clean_document'
            if reason is None:
                duplicate = dedup.add(text)
                if duplicate:
                    reason = 'clean_'+duplicate+'_overlap'
            if reason:
                stats[reason] += 1
            else:
                seen.add(canonical)
                row['source_content_sha256'] = row['content_sha256']
                row['content_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                row['text'] = text
                row['transform'] = 'prose-v1; inherited split; no factual rewriting'
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                ids = encode_text(tokenizer, text)
                assert 3 not in ids
                seq = np.array([BOS]+ids+[EOS], dtype='<u4')
                binary.write(seq.tobytes())
                offsets.append(offsets[-1]+len(seq))
                characters += len(text)
                retained.append(row['id'])
            audit.append(dict(id=row['id'], title=row.get('title'), url=row.get('url'),
                              before=before, after=text, kept=reason is None, reason=reason))
    dedup.close()
    np.save(output/'train.offsets.npy', np.array(offsets, dtype=np.int64))
    assert len(retained)>0
    manifest['splits']['train'] = dict(documents=len(retained), stored_tokens=offsets[-1],
                                      target_tokens=offsets[-1]-len(retained), characters=characters, unk_tokens=0)
    for v in manifest['splits'].values():
        v['document_ratio'] = v['documents']/sum(s['documents'] for s in manifest['splits'].values())
        v['token_ratio'] = v['target_tokens']/sum(s['target_tokens'] for s in manifest['splits'].values())
    manifest['counts'] = dict(stats, retained_train_documents=len(retained))
    manifest['cleaning'] = 'prose-v1 on train only; validation/test/tokenizer byte-identical to source'
    manifest['split_grouping'] = dict(method='inherit frozen source split assignments; reject cleaned train docs with exact heldout paragraph >=400 chars', source_split_grouping=manifest['split_grouping'])
    manifest['experiment'] = dict(source=str(source), source_fingerprint=parent_fingerprint,
                                  tokenizer_policy='frozen from original train only',
                                  heldout_policy='original validation/test unchanged; overlap used only for leakage exclusion',
                                  cleaner_sha256=sha256(Path(__file__).resolve().parents[1]/'minigpt/cleaning.py'))
    for name in list(manifest['hashes']):
        if name != 'source':
            manifest['hashes'][name] = sha256(output/name)
    write_json(output/'manifest.json', manifest)
    verify_artifacts(output)
    rng = random.Random(20261105)
    kept = [r for r in audit if r['kept']]
    rejected = [r for r in audit if not r['kept']]
    write_json(evidence/'sample_50_kept.json', rng.sample(kept, min(50,len(kept))))
    write_json(evidence/'sample_20_rejected.json', rng.sample(rejected,min(20,len(rejected))))
    write_json(evidence/'data_manifest.json', manifest)
    write_json(evidence/'cleaning_counts.json', dict(stats, kept_documents=len(retained), train_tokens=manifest['splits']['train']['target_tokens']))
    write_json(evidence/'heldout_verification.json', {name:sha256(source/name)==sha256(output/name) for name in manifest['hashes'] if name.startswith(('validation.','test.','tokenizer.'))})
    print(json.dumps(dict(stats, kept_documents=len(retained), train_tokens=manifest['splits']['train']['target_tokens']), indent=2))


if __name__ == '__main__':
    main()
