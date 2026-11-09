"""Exploratory secondary evaluation; never selects or replaces checkpoints."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from minigpt.cleaning import clean_prose
from minigpt.common import BOS, EOS, sha256, write_json
from minigpt.data import BlockDataset, iter_jsonl, encode_text, simhash
from minigpt.engine import load_neural, evaluate_neural
from minigpt.experiments import checked_result
from scripts.prepare_clean_control import paragraph_keys


def main():
    torch.set_num_threads(2)
    original=Path('data/wiki_main_20261102_v2')
    clean=Path('data/wiki_prose_control_20261105')
    probe=Path('data/wiki_prose_probe_20261105')
    probe.mkdir(parents=True,exist_ok=False)
    evidence=Path('artifacts/clean_prose_20261105')
    train_keys,train_sigs=set(),[]
    for root in [original,clean]:
        con=sqlite3.connect(f'file:{(root/"dedup.sqlite").as_posix()}?mode=ro',uri=True)
        signatures={key:int(sig) for key,sig in con.execute('select exact,sim from docs')}
        con.close()
        for row in iter_jsonl(root/'train.jsonl'):
            train_keys.update(paragraph_keys(row['text']))
            key=hashlib.sha256(' '.join(row['text'].casefold().split()).encode()).hexdigest()
            train_sigs.append(signatures[key])
    train_sigs=list(set(train_sigs))
    tok=Tokenizer.from_file(str(original/'tokenizer.json'))
    offsets,ids_kept,dropped=[0],[],[]
    with open(probe/'test.bin','wb') as stream:
        for row in iter_jsonl(original/'test.jsonl'):
            text,_=clean_prose(row['text'])
            if len(text)<600 or len(text)<.2*len(row['text']):
                dropped.append(dict(id=row['id'],reason='same prose filter')); continue
            sig=simhash(text)
            if paragraph_keys(text)&train_keys or any((sig^s).bit_count()<=3 for s in train_sigs):
                dropped.append(dict(id=row['id'],reason='train overlap after transformation')); continue
            seq=np.array([BOS]+encode_text(tok,text)+[EOS],dtype='<u4')
            stream.write(seq.tobytes())
            offsets.append(offsets[-1]+len(seq))
            ids_kept.append(row['id'])
    np.save(probe/'test.offsets.npy',np.array(offsets,dtype=np.int64))
    assert ids_kept
    rows=[]
    for data,run in [(original,Path('runs/large_20261105/gpt')),(clean,Path('runs/clean_prose_20261105_retry/gpt'))]:
        ev,_=checked_result(run)
        model,state=load_neural(run/'best.pt',torch.device('cuda'))
        assert state['data_fingerprint']==sha256(data/'manifest.json')
        result=evaluate_neural(model,BlockDataset(probe,'test',model.context),8,torch.device('cuda'))
        assert result['tokens']==offsets[-1]-len(ids_kept)
        assert sha256(run/'best.pt')==ev['checkpoint_sha256']
        rows.append(dict(run=str(run),checkpoint_sha256=ev['checkpoint_sha256'],**result))
        del model
        torch.cuda.empty_cache()
    report=dict(scope='exploratory secondary prose view; introduced after seeing original validation trend; no checkpoint reselection or rule tuning; report primary results too',
                source_test_fingerprint=sha256(original/'test.jsonl'),cleaner_sha256=sha256('minigpt/cleaning.py'),
                kept_ids=ids_kept,dropped=dropped,documents=len(ids_kept),test_tokens=offsets[-1]-len(ids_kept),
                probe_hashes={n:sha256(probe/n) for n in ['test.bin','test.offsets.npy']},results=rows)
    write_json(evidence/'secondary_prose_evaluation.json',report)
    print(json.dumps(dict(documents=len(ids_kept),tokens=report['test_tokens'],results=rows),indent=2))


if __name__=='__main__':
    main()
