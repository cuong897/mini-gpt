"""Paired 5M-token continuation experiment; only peak LR differs between arms."""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import BOS, environment, read_json, seed_all, sha256, write_json
from minigpt.data import encode_text, verify_artifacts
from minigpt.engine import evaluate_run, load_neural, train
from minigpt.generation import generate
from scripts.diagnose_conditioning import decode_ids

OUT = Path('artifacts/lr_control_20261105')
DATA = Path('data/wiki_main_20261102_v2')
PARENT = Path('runs/continuation_20261105/gpt')
ARMS = {'low': ('runs/lr_control_20261105/low/gpt', 3e-5),
        'high': ('runs/lr_control_20261105/high/gpt', 1e-4)}


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else:
        assert a == b


def prepare():
    if OUT.exists() or any(Path(run).exists() for run, _ in ARMS.values()):
        raise ValueError('Use new output paths; never overwrite an experiment')
    OUT.mkdir(parents=True)
    fingerprint = verify_artifacts(DATA)
    parent = torch.load(PARENT/'last.pt', map_location='cpu', weights_only=True)
    summary = read_json(PARENT/'summary.json')
    assert parent['data_fingerprint'] == fingerprint
    assert parent['tokens_seen'] == 20_000_000 and summary['cumulative_training_tokens'] == 40_000_000
    configs = {}
    for name, (run, lr) in ARMS.items():
        cfg = copy.deepcopy(parent['config'])
        cfg.update(learning_rate=lr, warmup_steps=100, max_tokens=5_000_000,
                   max_steps=20000, eval_every=100, log_every=50)
        write_json(OUT/f'{name}_config.json', cfg)
        configs[name] = cfg
    assert {k for k in configs['low'] if configs['low'][k] != configs['high'][k]} == {'learning_rate'}
    prompts = ['Việt Nam là một'] + read_json('configs/prompts.json')
    protocol = dict(
        scope='Paired exploratory continuation, one seed; no test scoring or tuning.',
        source_checkpoint=str((PARENT/'last.pt').resolve()), source_sha256=sha256(PARENT/'last.pt'),
        baseline_checkpoint_sha256=sha256(PARENT/'best.pt'), data_fingerprint=fingerprint,
        parent_cumulative_tokens=40_000_000, added_target_tokens_per_arm=5_000_000,
        config_difference=['learning_rate'], configs=configs,
        initialization='Identical model/optimizer/scaler/RNG/sampler from same last.pt. Reset stage and cosine schedule; both warm up for 100 steps.',
        checkpoint_selection='Best full original validation NLL within each completed fixed-budget arm; also retain/evaluate last.pt. No test evaluation.',
        prompts=prompts,
        sampling=[dict(temperature=0., seed=42), dict(temperature=.8, seed=42), dict(temperature=.8, seed=7)],
        generation=dict(max_new_tokens=80,top_k=40,top_p=.9),
        qualitative_rubric=dict(on_topic={'0':'contradicts or abandons prompt','1':'partial/key words only','2':'sustains subject'},
                                coherence={'0':'incoherent or degenerative loop','1':'some meaningful sentences with failures','2':'connected comprehensible passage'},
                                grammar={'0':'frequent broken clauses','1':'readable with errors','2':'mostly well formed'},
                                acceptable='on_topic=2 AND coherence=2 AND grammar>=1',
                                reviewer='assistant, not independent/blinded human benchmark; record every sample and a reason'),
        deployment_gate=dict(validation='strictly better than baseline', default_acceptable_at_least=8,
                             default_total=11, all_acceptable_at_least=17, all_total=33,
                             repetition='mean repeated 4gram fraction at T=.8 seed42 no worse than baseline+0.01',
                             policy='Only consider replacing demo if all gates pass. Lower loss alone is insufficient.'),
        limits=['Single seed, short additional budget; does not establish an optimal LR.',
                'Selected validation loss is exploratory; prompts have been inspected previously.',
                'Source integrity checks hash test artifacts but never score test.'],
        code_hashes={p:sha256(p) for p in ['minigpt/engine.py','minigpt/models.py','minigpt/generation.py',
                                         'minigpt/data.py','scripts/diagnose_conditioning.py',__file__]})
    write_json(OUT/'protocol.json', protocol)
    write_json(OUT/'environment.json', environment())
    del parent
    print('Protocol frozen: two arms, 5,000,000 targets each; only peak LR differs.', flush=True)


def initialize():
    protocol = read_json(OUT/'protocol.json')
    assert sha256(PARENT/'last.pt') == protocol['source_sha256']
    for name, (run, _) in ARMS.items():
        train(OUT/f'{name}_config.json', DATA, run, 'cuda', continue_from=PARENT/'last.pt', stop_after=0)
    a = torch.load(PARENT/'last.pt', map_location='cpu', weights_only=True)
    initial_hashes = {}
    for name, (run, _) in ARMS.items():
        b = torch.load(Path(run)/'last.pt', map_location='cpu', weights_only=True)
        for key in ['model_state','optimizer_state','scaler_state','rng','sampler_rng']:
            assert_equal(a[key], b[key])
        assert b['step'] == b['tokens_seen'] == 0
        lineage = read_json(Path(run)/'lineage.json')
        assert lineage['source_sha256'] == protocol['source_sha256']
        assert lineage['prior_training_tokens'] == 40_000_000
        initial_hashes[name] = sha256(Path(run)/'last.pt')
        del b
    write_json(OUT/'initialization_verification.json',dict(exact_parent_model_optimizer_scaler_rng_sampler=True,
               stage_steps_zero=True,initial_checkpoint_hashes=initial_hashes,source_sha256=protocol['source_sha256']))
    print('Both initial states exactly match parent weights, optimizer, scaler, RNG and sampler.', flush=True)


def run_arm(name):
    run = Path(ARMS[name][0])
    train(OUT/f'{name}_config.json', DATA, run, 'cuda', resume=run/'last.pt')


def verify_pair():
    protocol = read_json(OUT/'protocol.json')
    summaries = {}
    histories = {}
    states = []
    for name, (run, _) in ARMS.items():
        run = Path(run)
        cfg = read_json(run/'config.json')
        assert cfg == protocol['configs'][name]
        summary = read_json(run/'summary.json')
        assert summary['budget_completed'] and summary['tokens_seen'] == 5_000_000
        assert summary['cumulative_training_tokens'] == 45_000_000
        lineage = read_json(run/'lineage.json')
        assert lineage['source_sha256'] == protocol['source_sha256']
        summaries[name] = summary
        histories[name] = [json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        state = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        states.append({k:state[k] for k in ['step','tokens_seen','sampler_rng','rng']})
        del state
    assert summaries['low']['steps'] == summaries['high']['steps']
    assert [r['tokens_seen'] for r in histories['low']] == [r['tokens_seen'] for r in histories['high']]
    assert_equal(states[0], states[1])
    assert sha256(PARENT/'last.pt') == protocol['source_sha256']
    assert sha256(PARENT/'best.pt') == protocol['baseline_checkpoint_sha256']
    assert verify_artifacts(DATA) == protocol['data_fingerprint']
    write_json(OUT/'pair_verification.json',dict(only_lr_differs=True,identical_initial_states=True,
        identical_per_step_token_counts=True,identical_final_rng_and_sampler=True,
        original_checkpoints_unchanged=True,target_tokens_each=5_000_000,steps_each=summaries['low']['steps'],summaries=summaries))


def samples():
    verify_pair()
    protocol = read_json(OUT/'protocol.json')
    tokenizer = Tokenizer.from_file(str(DATA/'tokenizer.json'))
    device = torch.device('cuda')
    rows, metrics = [], []
    for name, run in [('parent',str(PARENT))]+[(n,v[0]) for n,v in ARMS.items()]:
        run = Path(run)
        digest = sha256(run/'best.pt')
        ev = evaluate_run(DATA,run,'validation','cuda',batch_size=8)
        model,state = load_neural(run/'best.pt',device)
        assert state['data_fingerprint'] == protocol['data_fingerprint']
        metrics.append(dict(arm=name,run=str(run),checkpoint_sha256=digest,
                            checkpoint_step=state['step'],checkpoint_stage_tokens=state['tokens_seen'],
                            selected_validation_nll=ev['loss'],selected_validation_perplexity=ev['perplexity']))
        for i,prompt in enumerate(protocol['prompts']):
            for settings in protocol['sampling']:
                seed_all(settings['seed'])
                ids = decode_ids(model,[BOS]+encode_text(tokenizer,prompt),device,count=80,**settings)
                fours=list(zip(ids,ids[1:],ids[2:],ids[3:]))
                repeated=sum(n-1 for n in Counter(fours).values())/len(fours) if fours else 0.
                rows.append(dict(arm=name,prompt_id=i,prompt=prompt,continuation=tokenizer.decode(ids),generated_ids=ids,
                                 checkpoint_sha256=digest,**settings,top_k=40,top_p=.9,max_new_tokens=80,
                                 repeated_4gram_fraction=repeated))
            write_json(OUT/'samples.json',rows)
            print(f'Samples {name}: {i+1}/11',flush=True)
        del model
        torch.cuda.empty_cache()
        public=generate(DATA,run,protocol['prompts'][0],max_new_tokens=80,seed=42,temperature=.8,top_k=40,top_p=.9,device_name='cuda')
        expected=next(r for r in rows if r['arm']==name and r['prompt_id']==0 and r['temperature']==.8 and r['seed']==42)
        assert public['generated_ids']==expected['generated_ids']
        write_json(OUT/f'public_generation_{name}.json',public)
        assert sha256(run/'best.pt')==digest
        write_json(OUT/'metrics.json',metrics)
    old=read_json('artifacts/conditioning_20261105/live_samples.json')
    parent=[r for r in rows if r['arm']=='parent']
    assert len(parent)==len(old)==33
    for a,b in zip(parent,old):
        assert a['prompt']==b['prompt'] and a['temperature']==b['temperature'] and a['seed']==b['seed']
        assert a['generated_ids']==b['generated_ids']
    write_json(OUT/'generation_verification.json',dict(all_parent_33_outputs_reproduced=True,
               public_generate_matches_all_arms=True,all_samples=99))
    lines=['# Toàn bộ mẫu đối chứng learning rate', '', 'Cùng prompt, seed, k40/p0,9 và 80 token. Parent40M; low/high được thêm5M mỗi lượt. Chọn checkpoint theo validation.', '']
    for i,prompt in enumerate(protocol['prompts']):
        lines += [f'## {i+1}. {prompt}', '']
        for row in (r for r in rows if r['prompt_id']==i):
            lines += [f"### {row['arm']} · T={row['temperature']} · seed={row['seed']}",'',row['prompt']+row['continuation'],'']
    (OUT/'samples.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('phase',choices=['prepare','initialize','low','high','samples'])
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    seed_all(42)
    if args.phase=='prepare': prepare()
    elif args.phase=='initialize': initialize()
    elif args.phase in ARMS: run_arm(args.phase)
    else: samples()


if __name__=='__main__': main()
