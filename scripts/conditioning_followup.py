"""Exploratory follow-up after context and memorization diagnostics, no test split."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import BOS, PAD, read_json, sha256, write_json
from minigpt.data import BlockDataset, encode_text, iter_jsonl, verify_artifacts
from minigpt.engine import load_neural
from minigpt.models import token_loss
from scripts.diagnose_conditioning import OUT, SOURCE, LIVE_RUN, SMALL, RUN


@torch.no_grad()
def fit(model, root, split, device):
    data = BlockDataset(root, split, model.context)
    total_nll = good = count = 0
    for i, (x, y) in enumerate(torch.utils.data.DataLoader(data, batch_size=8)):
        x,y=x.to(device),y.to(device)
        logits=model(x)
        total_nll += token_loss(logits,y,reduction='sum').item()
        count += int(y.ne(PAD).sum())
        good += int((logits.argmax(-1).eq(y)&y.ne(PAD)).sum())
        if (i+1)%1000 == 0:
            print(f'{split}: {count:,} targets evaluated',flush=True)
    assert count==read_json(root/'manifest.json')['splits'][split]['target_tokens']
    return dict(loss=total_nll/count, tokens=count, accuracy=good/count, correct=good)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    protocol=read_json(OUT/'protocol.json')
    assert sha256(LIVE_RUN/'best.pt')==protocol['live_checkpoint_sha256']
    write_json(OUT/'followup_protocol.json',dict(
        timing='Introduced after observing initial diagnostics, before the following model measurements',
        scope='Full original train vs validation fit; immediate next-token rank; lexical corpus counts. No test, no tuning.',
        purpose='Distinguish an obvious train/validation gap from prompt decoding errors; no claim of a unique root cause',
        script_sha256=sha256(__file__), checkpoint_sha256=protocol['live_checkpoint_sha256']))
    device=torch.device('cuda')
    model,state=load_neural(LIVE_RUN/'best.pt',device)
    assert state['data_fingerprint']==verify_artifacts(SOURCE)
    tok=Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
    prompt='Việt Nam là một'
    prefix=[BOS]+encode_text(tok,prompt)
    with torch.no_grad():
        logits=model(torch.tensor([prefix],device=device))[0,-1].float().cpu()
    probs=logits.softmax(-1)
    candidate=encode_text(tok,prompt+' quốc gia ở Đông Nam Á.')
    assert candidate[:len(prefix)-1]==prefix[1:]
    token=candidate[len(prefix)-1]
    # Reproduce support of the existing decoder without drawing any samples.
    filtered=logits.double().clone()
    filtered[[0,1,3]]=-float('inf')
    filtered=(filtered-filtered.max())/.8
    values,indices=filtered.topk(40)
    support=torch.full_like(filtered,-float('inf'))
    support[indices]=values
    sorted_logits,order=support.sort(descending=True)
    probabilities=sorted_logits.softmax(-1)
    remove=(probabilities.cumsum(-1)-probabilities)>=.9
    support[order[remove]]=-float('inf')
    next_token=dict(prompt=prompt, intended_next_token=tok.decode([token]),token_id=token,
                    raw_probability=float(probs[token]),rank=1+int((logits>logits[token]).sum()),
                    survives_top_k40_top_p09_temperature08=bool(torch.isfinite(support[token])),
                    filtered_probability=float(support.softmax(-1)[token]))
    write_json(OUT/'next_token_failure.json',next_token)
    result={}
    for split in ['train','validation']:
        result[split]=fit(model,SOURCE,split,device)
        write_json(OUT/'main_fit.json',result)
        print(split,json.dumps(result[split]),flush=True)
    result['validation_minus_train_loss']=result['validation']['loss']-result['train']['loss']
    write_json(OUT/'main_fit.json',result)
    del model
    torch.cuda.empty_cache()
    # Check where the deliberate memorization model makes its remaining errors.
    model,_=load_neural(RUN/'last.pt',device)
    errors=[]
    with torch.no_grad():
        for index,(x,y) in enumerate(BlockDataset(SMALL,'train',256)):
            predicted=model(x[None].to(device))[0].argmax(-1).cpu()
            positions=((predicted!=y)&y.ne(PAD)).nonzero().flatten().tolist()
            errors.append(dict(document_index=index,error_positions=positions))
    write_json(OUT/'micro_error_positions.json',errors)
    patterns={'taxonomic_intro':r'\b(?:là|gồm|chứa) (?:một |1 )?(?:loài|chi|họ|giống|phân loài)\b',
              'film_music_intro':r'\b(?:bộ phim|ca sĩ|nhóm nhạc|album|đĩa đơn)\b'}
    rows=list(iter_jsonl(SOURCE/'train.jsonl'))
    offsets=np.load(SOURCE/'train.offsets.npy')
    targets=np.diff(offsets)-1
    audit={}
    for name,pattern in patterns.items():
        indices=[i for i,r in enumerate(rows) if re.search(pattern,r['text'][:500],re.I)]
        audit[name]=dict(regex=pattern,documents=len(indices),document_fraction=len(indices)/len(rows),
                         full_document_target_token_fraction=float(targets[indices].sum()/targets.sum()),
                         sample_titles=[rows[i].get('title') for i in indices[:12]])
    write_json(OUT/'corpus_lexical_audit.json',dict(
        scope='Exploratory, overlapping lexical indicators in first 500 characters; NOT a validated topic classifier or proof of causality',
        documents=len(rows),indicators=audit,
        source_sampling='Existing fetch uses filter, streaming shuffle(buffer_size=1000), then first 10000; bounded-buffer shuffle is not a uniform sample of the entire stream. No new data collected.'))
    assert sha256(LIVE_RUN/'best.pt')==protocol['live_checkpoint_sha256']
    print(json.dumps(dict(next_token=next_token,fit=result,lexical=audit),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
