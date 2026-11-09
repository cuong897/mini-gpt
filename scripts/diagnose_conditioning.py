"""Bounded diagnostic, not a new main experiment or a replacement demo model.

Use only original train/validation. Freeze probes before measuring the checkpoint.
The small real-text run deliberately measures memorization with the production trainer.
"""
import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.cleaning import clean_prose
from minigpt.common import BOS, EOS, PAD, environment, read_json, seed_all, sha256, write_json
from minigpt.data import BlockDataset, encode_text, iter_jsonl, verify_artifacts
from minigpt.engine import evaluate_neural, load_neural, train
from minigpt.generation import generate, sample_token
from minigpt.models import build_model

SOURCE = Path('data/wiki_main_20261102_v2')
CLEAN = Path('data/wiki_prose_control_20261105')
LIVE_RUN = Path('runs/continuation_20261105/gpt')
OUT = Path('artifacts/conditioning_20261105')
SMALL = Path('data/conditioning_micro_20261105')
RUN = Path('runs/conditioning_micro_20261105/gpt')
TRAIN_IDS = ['3113968', '3120947', '2611762', '2993747', '3120499', '2611792',
             '3121398', '3123080', '3120402', '3102530', '3116573', '2889696']
# Authored diagnostic pairs, NOT corpus data, test benchmarks, or fact-checking evidence.
PAIRS = [
    ('Việt Nam là một', 'Cá rô phi là một',
     ' quốc gia ở Đông Nam Á.', ' loài cá sống trong nước ngọt.'),
    ('Hà Nội là thủ đô của', 'Tokyo là thủ đô của', ' Việt Nam.', ' Nhật Bản.'),
    ('Người nông dân đang', 'Người bác sĩ đang',
     ' trồng lúa trên cánh đồng.', ' khám bệnh cho bệnh nhân.'),
    ('Con mèo đang', 'Cái cây đang', ' bắt chuột trong nhà.', ' mọc thêm nhiều lá mới.'),
    ('Học sinh đến trường để', 'Bệnh nhân đến bệnh viện để',
     ' học tập cùng thầy cô.', ' được bác sĩ khám bệnh.'),
    ('Con chim dùng cánh để', 'Con cá dùng vây để', ' bay trên bầu trời.', ' bơi dưới nước.'),
    ('Mặt Trời mọc ở hướng', 'Mặt Trời lặn ở hướng', ' đông.', ' tây.'),
    ('Thư viện là nơi', 'Sân bay là nơi',
     ' lưu trữ sách và phục vụ người đọc.', ' máy bay cất cánh và hạ cánh.'),
]


@torch.no_grad()
def suffix_scores(model, prefix, suffix, device):
    """Score only suffix targets, preserving their exact position and token IDs."""
    assert prefix and suffix and len(prefix) + len(suffix) - 1 <= model.context
    ids = torch.tensor([prefix + suffix], device=device)
    logits = model(ids[:, :-1])[0, len(prefix)-1:].float()
    targets = ids[0, len(prefix):]
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    correct = logits.argmax(-1).eq(targets)
    return dict(nll=float(losses.mean()), sum_nll=float(losses.sum()), tokens=len(suffix),
                first_nll=float(losses[0]), first8_nll=float(losses[:8].mean()),
                accuracy=float(correct.float().mean()))


def text_suffix(tokenizer, prefix, suffix):
    p = encode_text(tokenizer, prefix)
    full = encode_text(tokenizer, prefix + suffix)
    if full[:len(p)] != p:
        raise ValueError('BPE merges across scoring boundary; choose a word boundary')
    return [BOS]+p, full[len(p):]


@torch.no_grad()
def decode_ids(model, prefix, device, count=32, temperature=0., seed=42):
    ids, produced = list(prefix), []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(count):
        logits = model(torch.tensor([ids[-model.context:]], device=device))[0, -1].cpu()
        t = sample_token(logits, generator, temperature=temperature, top_k=40, top_p=.9)
        if t == EOS:
            break
        ids.append(t)
        produced.append(t)
    return produced


def safe_prefix(tokenizer, text):
    ids = encode_text(tokenizer, text)
    for match in re.finditer(r'\s+', text):
        prefix = text[:match.start()]
        p = encode_text(tokenizer, prefix)
        if 24 <= len(p) <= 45 and ids[:len(p)] == p and len(ids)-len(p) >= 32:
            return prefix, [BOS]+p, ids[len(p):len(p)+32]
    raise ValueError('No valid prefix boundary')


def prepare():
    if OUT.exists() or SMALL.exists() or RUN.exists():
        raise ValueError('Diagnostic outputs already exist; do not overwrite evidence')
    OUT.mkdir(parents=True)
    SMALL.mkdir(parents=True)
    original = verify_artifacts(SOURCE)
    verify_artifacts(CLEAN)
    tok = Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
    selected = {str(r['id']): r for r in iter_jsonl(CLEAN/'train.jsonl') if str(r['id']) in TRAIN_IDS}
    train_rows = []
    for key in TRAIN_IDS:
        row = copy.deepcopy(selected[key])
        row['text'] = row['text'].split('\n\n')[0]
        row['diagnostic_note'] = 'Real train excerpt; reviewed for readability, not independently fact-checked'
        row['content_sha256'] = __import__('hashlib').sha256(row['text'].encode()).hexdigest()
        assert 70 <= len(encode_text(tok, row['text'])) <= 180
        safe_prefix(tok, row['text'])
        train_rows.append(row)
    validation = []
    for row in iter_jsonl(SOURCE/'validation.jsonl'):
        text, _ = clean_prose(row['text'])
        paragraph = text.split('\n\n')[0]
        if 70 <= len(encode_text(tok, paragraph)) <= 180 and paragraph.endswith('.'):
            validation.append(dict(row, text=paragraph))
        if len(validation) == 64:
            break
    assert len(validation) == 64
    # Validation excerpts are diagnostic, inherited from original validation only.
    assert not {r['id'] for r in train_rows} & {r['id'] for r in validation}
    shutil.copyfile(SOURCE/'tokenizer.json', SMALL/'tokenizer.json')
    hashes, splits = {'tokenizer.json': sha256(SMALL/'tokenizer.json')}, {}
    for split, rows in [('train', train_rows), ('validation', validation)]:
        offsets = [0]
        with open(SMALL/f'{split}.jsonl', 'w', encoding='utf-8') as out, open(SMALL/f'{split}.bin', 'wb') as binary:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False)+'\n')
                seq = np.array([BOS]+encode_text(tok, row['text'])+[EOS], dtype='<u4')
                binary.write(seq.tobytes())
                offsets.append(offsets[-1]+len(seq))
        np.save(SMALL/f'{split}.offsets.npy', np.array(offsets, dtype=np.int64))
        for ext in ['jsonl', 'bin', 'offsets.npy']:
            hashes[f'{split}.{ext}'] = sha256(SMALL/f'{split}.{ext}')
        splits[split] = dict(documents=len(rows), stored_tokens=offsets[-1], target_tokens=offsets[-1]-len(rows))
    write_json(SMALL/'manifest.json', dict(kind='real', vocab_size=tok.get_vocab_size(), hashes=hashes,
               splits=splits, provenance=read_json(SOURCE/'manifest.json')['provenance'],
               diagnostic_only=True, source_fingerprint=original,
               policy='Inherited train/validation; excerpt only; frozen original train tokenizer; no test'))
    config = read_json('configs/gpt_large.json')
    config.update(kind='diagnostic', batch_size=4, accumulation=1, max_steps=800, max_tokens=None,
                  warmup_steps=20, eval_every=100, log_every=50)
    write_json(OUT/'micro_config.json', config)
    prompts = ['Việt Nam là một'] + read_json('configs/prompts.json')
    protocol = dict(scope='Debugging evidence only. Not a quality benchmark, not main training, no test access.',
                    source_fingerprint=original, live_checkpoint=str(LIVE_RUN/'best.pt'),
                    live_checkpoint_sha256=sha256(LIVE_RUN/'best.pt'), script_sha256=sha256(__file__),
                    context_ablation='64 validation excerpts; 32 prefix tokens + BOS, same 32 suffix tokens; cyclic mismatched prefix, equal length and positions',
                    contrast_pairs=PAIRS, contrasts='Each suffix compared under both prompts; raw likelihood and length-normalized scores retained; diagnostic pairs authored before scoring',
                    prompts=prompts, sampling=[dict(temperature=0., seed=42), dict(temperature=.8, seed=42), dict(temperature=.8, seed=7)],
                    generation_tokens=80, micro_train_ids=TRAIN_IDS, micro_config=config,
                    micro_checkpoint='last.pt at fixed step 800, not best validation; deliberate train memorization',
                    micro_pass_criteria=dict(train_teacher_forced_accuracy_at_least=.95,
                                             exact_32token_greedy_continuations_at_least=10,
                                             total_continuations=12),
                    limits=['One seed; 12 reviewed excerpts do not characterize the full corpus.',
                            'Memorization is not generalization. Never deploy this diagnostic checkpoint.',
                            'Prompt likelihood/sensitivity does not establish coherent generation or factual knowledge.',
                            'Validation was already used for checkpoint selection; these are exploratory diagnostics.'])
    write_json(OUT/'protocol.json', protocol)
    write_json(OUT/'environment.json', environment())
    write_json(OUT/'train_excerpts.json', train_rows)
    print('Protocol frozen; 12 train excerpts, 64 validation excerpts; no test.', flush=True)


def live_diagnostics():
    protocol = read_json(OUT/'protocol.json')
    assert sha256(LIVE_RUN/'best.pt') == protocol['live_checkpoint_sha256']
    tok = Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
    device = torch.device('cuda')
    model, state = load_neural(LIVE_RUN/'best.pt', device)
    assert state['data_fingerprint'] == protocol['source_fingerprint']
    excerpts = list(iter_jsonl(SMALL/'validation.jsonl'))
    seqs = [encode_text(tok, r['text']) for r in excerpts]
    ablations = []
    for i, (row, ids) in enumerate(zip(excerpts, seqs)):
        own = suffix_scores(model, [BOS]+ids[:32], ids[32:64], device)
        wrong = suffix_scores(model, [BOS]+seqs[(i+1)%len(seqs)][:32], ids[32:64], device)
        ablations.append(dict(id=row['id'], title=row.get('title'), prefix=tok.decode(ids[:32]),
                              mismatched_prefix=tok.decode(seqs[(i+1)%len(seqs)][:32]), suffix=tok.decode(ids[32:64]),
                              correct=own, mismatched=wrong))
    write_json(OUT/'context_ablation.json', dict(rows=ablations,
        mean_correct_nll=float(np.mean([r['correct']['nll'] for r in ablations])),
        mean_mismatched_nll=float(np.mean([r['mismatched']['nll'] for r in ablations])),
        mean_first8_correct_nll=float(np.mean([r['correct']['first8_nll'] for r in ablations])),
        mean_first8_mismatched_nll=float(np.mean([r['mismatched']['first8_nll'] for r in ablations])),
        correct_prefix_wins=sum(r['correct']['nll'] < r['mismatched']['nll'] for r in ablations), total=len(ablations)))
    pairs = []
    for p1, p2, s1, s2 in PAIRS:
        scores = [[suffix_scores(model, *text_suffix(tok, p, s), device) for s in [s1,s2]] for p in [p1,p2]]
        # Same completion under different contexts controls completion length/frequency.
        margins = [scores[1][0]['nll']-scores[0][0]['nll'], scores[0][1]['nll']-scores[1][1]['nll']]
        pairs.append(dict(prompts=[p1,p2], suffixes=[s1,s2], scores=scores,
                          matched_context_advantage=margins,
                          preferred_suffix_by_mean_nll=[int(scores[i][1]['nll'] < scores[i][0]['nll']) for i in [0,1]]))
    write_json(OUT/'contrast_scores.json', dict(rows=pairs,
        positive_context_margins=sum(m>0 for r in pairs for m in r['matched_context_advantage']), total=2*len(pairs)))
    # Vocabulary views show model probabilities, not a claim that top tokens are whole words.
    with torch.no_grad():
        tops = []
        for prompt in protocol['prompts']:
            ids = [BOS]+encode_text(tok,prompt)
            probs = model(torch.tensor([ids], device=device))[0,-1].softmax(-1)
            values, indices = probs.topk(15)
            tops.append(dict(prompt=prompt, top_tokens=[dict(id=int(t), text=tok.decode([int(t)]), probability=float(p)) for t,p in zip(indices,values)]))
    write_json(OUT/'next_token_top15.json', tops)
    samples = []
    for prompt in protocol['prompts']:
        for settings in protocol['sampling']:
            ids = decode_ids(model, [BOS]+encode_text(tok,prompt), device, count=80, **settings)
            samples.append(dict(prompt=prompt, continuation=tok.decode(ids), generated_ids=ids, **settings))
        print('Prompt diagnostic: '+prompt, flush=True)
        write_json(OUT/'live_samples.json', samples)
    del model
    torch.cuda.empty_cache()
    # Cross-check custom diagnostic decoder against the actual public generator.
    public = generate(SOURCE,LIVE_RUN,protocol['prompts'][0],max_new_tokens=80,seed=42,
                      temperature=.8,top_k=40,top_p=.9,device_name='cuda')
    expected = next(s for s in samples if s['prompt']==protocol['prompts'][0] and s['temperature']==.8 and s['seed']==42)
    assert public['generated_ids']==expected['generated_ids']
    write_json(OUT/'public_generation_check.json', public)
    assert sha256(LIVE_RUN/'best.pt') == protocol['live_checkpoint_sha256']


@torch.no_grad()
def train_accuracy(model, dataset, device):
    good = total = 0
    model.eval()
    for x,y in torch.utils.data.DataLoader(dataset,batch_size=4):
        x,y = x.to(device),y.to(device)
        mask = y.ne(PAD)
        good += int((model(x).argmax(-1).eq(y)&mask).sum())
        total += int(mask.sum())
    return dict(correct=good,tokens=total,accuracy=good/total)


def micro_result(checkpoint=None):
    tok = Tokenizer.from_file(str(SMALL/'tokenizer.json'))
    config = read_json(OUT/'micro_config.json')
    device = torch.device('cuda')
    seed_all(config['seed'])
    if checkpoint:
        model, state = load_neural(RUN/checkpoint, device)
        assert state['data_fingerprint'] == verify_artifacts(SMALL)
        assert state['step'] == config['max_steps']
    else:
        model = build_model(config, tok.get_vocab_size()).to(device).eval()
    result = dict(train=evaluate_neural(model,BlockDataset(SMALL,'train',256),4,device),
                  validation=evaluate_neural(model,BlockDataset(SMALL,'validation',256),4,device),
                  teacher_forced=train_accuracy(model,BlockDataset(SMALL,'train',256),device))
    if checkpoint:
        samples=[]
        for row in iter_jsonl(SMALL/'train.jsonl'):
            prompt,prefix,expected = safe_prefix(tok,row['text'])
            produced = decode_ids(model,prefix,device)
            match=0
            for a,b in zip(expected,produced):
                if a!=b: break
                match+=1
            samples.append(dict(id=row['id'],prompt=prompt,expected=tok.decode(expected),
                                continuation=tok.decode(produced),expected_ids=expected,generated_ids=produced,
                                exact_match=produced==expected,matching_prefix_tokens=match))
        result.update(checkpoint_sha256=sha256(RUN/checkpoint),samples=samples,
                      exact_32token_continuations=sum(r['exact_match'] for r in samples))
        result['memorization_pass'] = (result['teacher_forced']['accuracy']>=.95 and result['exact_32token_continuations']>=10)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('phase',choices=['prepare','live','micro','report'])
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    seed_all(42)
    if args.phase=='prepare': prepare()
    elif args.phase=='live': live_diagnostics()
    elif args.phase=='micro':
        write_json(OUT/'micro_before.json',micro_result())
        train(OUT/'micro_config.json',SMALL,RUN,device_name='cuda')
        result=micro_result('last.pt')
        write_json(OUT/'micro_after.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
    else:
        lines=['# Chẩn đoán khả năng bám ngữ cảnh', '', 'Dữ liệu chẩn đoán: train/validation, không dùng test. Đây không phải bản demo đã sửa.', '']
        for name in ['context_ablation','contrast_scores','micro_before','micro_after']:
            result=read_json(OUT/f'{name}.json')
            lines += [f'## {name}', '', '```json',json.dumps({k:v for k,v in result.items() if k not in ['rows','samples']},ensure_ascii=False,indent=2),'```','']
        lines+=['## Toàn bộ mẫu từ checkpoint demo', '']
        for row in read_json(OUT/'live_samples.json'):
            lines += [f"### {row['prompt']} · T={row['temperature']} · seed={row['seed']}",'',row['prompt']+row['continuation'],'']
        lines+=['## Học thuộc 12 đoạn: kết quả và văn bản gốc', '']
        for row in read_json(OUT/'micro_after.json')['samples']:
            lines += [f"### {row['id']} · khớp {row['matching_prefix_tokens']}/32 token",'',f"Prompt: {row['prompt']}",'',f"Sinh: {row['continuation']}",'',f"Gốc: {row['expected']}",'']
        (OUT/'results.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
