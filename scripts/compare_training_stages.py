"""Report a continuation experiment, retaining all prompts and sampling modes."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import read_json, sha256, write_json
from minigpt.experiments import checked_result
from minigpt.generation import generate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lineage = read_json(Path(args.stage)/'lineage.json')
    assert lineage['source_sha256'] == sha256(Path(args.parent)/'last.pt')
    runs = [args.parent, args.stage]
    results, samples = [], []
    signature = None
    prompts = read_json('configs/prompts.json')
    for run in runs:
        ev, meta = checked_result(run)
        sig = tuple(ev[k] for k in ['data_fingerprint', 'context', 'protocol', 'tokens', 'parameters'])
        assert signature is None or signature == sig
        signature = sig
        assert meta['budget_completed'] and meta['checkpoint_binding_verified']
        total = meta.get('cumulative_training_tokens', meta['tokens_seen'])
        results.append(dict(run=run, cumulative_tokens=total,
                            stage_tokens=meta['tokens_seen'], validation_loss=meta['best_validation_loss'],
                            test_loss=ev['loss'], test_perplexity=ev['perplexity'],
                            stage_seconds=meta['elapsed_train_seconds'], checkpoint_sha256=ev['checkpoint_sha256']))
        for temperature in [.8, 1.0]:
            for index, prompt in enumerate(prompts):
                sample = generate(args.data, run, prompt, max_new_tokens=80, seed=42,
                                  temperature=temperature, top_k=40, top_p=.9, device_name='cuda')
                assert sample['checkpoint_sha256'] == ev['checkpoint_sha256']
                sample['run'] = run
                sample['prompt_id'] = index
                tokens = sample['generated_ids']
                pairs = list(zip(tokens, tokens[1:]))
                fours = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))
                sample['distinct_2'] = len(set(pairs))/len(pairs) if pairs else 0
                sample['repeated_4gram_fraction'] = (sum(n-1 for n in Counter(fours).values())/len(fours) if fours else 0)
                samples.append(sample)
                write_json(output/'samples.json', samples)
                print(f'{run}, T={temperature}: {index+1}/{len(prompts)}', flush=True)
    metrics = []
    for row in results:
        for temperature in [.8, 1.0]:
            subset = [s for s in samples if s['run']==row['run'] and s['sampling']['temperature']==temperature]
            metrics.append(dict(run=row['run'], temperature=temperature, samples=len(subset),
                                mean_distinct_2=sum(s['distinct_2'] for s in subset)/len(subset),
                                mean_repeated_4gram_fraction=sum(s['repeated_4gram_fraction'] for s in subset)/len(subset),
                                eos_count=sum(s['stop_reason']=='eos' for s in subset),
                                replacement_character_samples=sum('\ufffd' in s['continuation'] for s in subset)))
    write_json(output/'comparison.json', dict(results=results, sampling_metrics=metrics))
    lines = ['# GPT 29,4M: trước và sau giai đoạn học thêm', '',
             '| Tổng token train | Validation NLL | Test NLL | Test PPL | Thời gian riêng giai đoạn (s) |',
             '|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['cumulative_tokens']:,} | {r['validation_loss']:.4f} | {r['test_loss']:.4f} | {r['test_perplexity']:.3f} | {r['stage_seconds']:.2f} |")
    lines += ['', 'Cùng dữ liệu/tokenizer/kiến trúc; giai đoạn mới kế thừa optimizer/RNG/sampler, dùng lịch LR mới 3e-5 → 3e-6.',
              'Khác cả số token và lịch LR; không quy toàn bộ thay đổi cho riêng số token. Một seed, chưa chấm người độc lập.',
              'Chọn checkpoint theo validation; test chỉ báo cáo sau khi hoàn tất.', '',
              '## Chỉ số lặp theo token BPE', '',
              '| Tổng token train | T | Mẫu | Distinct-2 TB | Tỷ lệ 4-gram lặp TB | Dừng EOS | Mẫu có � |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        total = next(r['cumulative_tokens'] for r in results if r['run']==m['run'])
        lines.append(f"| {total:,} | {m['temperature']} | {m['samples']} | {m['mean_distinct_2']:.3f} | {m['mean_repeated_4gram_fraction']:.3f} | {m['eos_count']} | {m['replacement_character_samples']} |")
    lines += ['', 'Distinct/lặp chỉ đo mẫu token, không đo tính đúng hoặc mạch lạc.',
              'Cùng seed 42, k40/p0,9, tối đa 80 token; hai chế độ chỉ khác temperature.', '', '## Toàn bộ 40 mẫu', '']
    for index, prompt in enumerate(prompts):
        lines += [f'### {index+1}. {prompt}', '']
        for sample in (s for s in samples if s['prompt_id']==index):
            total = next(r['cumulative_tokens'] for r in results if r['run']==sample['run'])
            lines += [f"**{total:,} token train · T={sample['sampling']['temperature']}**", '',
                      sample['prompt']+sample['continuation'], '']
    (output/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    for run, label in zip(runs, ['Stage 1', 'Stage 2 (lower LR)']):
        history = [json.loads(l) for l in (Path(run)/'metrics.jsonl').read_text().splitlines()]
        history = [r for r in history if 'validation_loss' in r]
        offset = lineage['prior_training_tokens'] if run==args.stage else 0
        ax.plot([(r['tokens_seen']+offset)/1e6 for r in history],
                [r['validation_loss'] for r in history], 'o-', label=label)
    ax.set(xlabel='Cumulative training tokens (millions)', ylabel='Validation NLL', title='Measured validation loss; one seed')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'validation.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(results=results, sampling_metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
