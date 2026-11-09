"""Compare completed GPT runs and retain every fixed-prompt continuation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import read_json, write_json
from minigpt.experiments import checked_result
from minigpt.generation import generate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompts', default='configs/prompts.json')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [read_json(Path(run) / 'config.json') for run in args.runs]
    common = [{k: v for k, v in c.items() if k != 'architecture'} for c in configs]
    if any(c != common[0] for c in common):
        raise ValueError('Training settings differ beyond architecture')
    rows, samples = [], []
    signature = None
    prompts = read_json(args.prompts)
    settings = dict(max_new_tokens=80, seed=42, temperature=.8, top_k=40, top_p=.9)
    for run in args.runs:
        ev, meta = checked_result(run)
        current = (ev['data_fingerprint'], ev['protocol'], ev['context'], ev['tokens'])
        if signature is not None and current != signature:
            raise ValueError('Incompatible evaluation protocols/data')
        signature = current
        if (ev['model'] != 'gpt' or not meta['budget_completed']
                or meta['tokens_seen'] != configs[0]['max_tokens']
                or not meta['checkpoint_binding_verified']):
            raise ValueError('Requires completed, verified GPT checkpoints')
        rows.append(dict(run=run, parameters=meta['parameters'],
                         training_tokens=meta['tokens_seen'], steps=meta['steps'],
                         validation_loss=meta['best_validation_loss'],
                         test_loss=ev['loss'], test_perplexity=ev['perplexity'],
                         training_seconds=meta['elapsed_train_seconds'],
                         checkpoint_sha256=ev['checkpoint_sha256']))
        for index, prompt in enumerate(prompts):
            sample = generate(args.data, run, prompt, device_name='cuda', **settings)
            if sample['checkpoint_sha256'] != ev['checkpoint_sha256']:
                raise ValueError('Generation/evaluation checkpoints differ')
            sample['prompt_id'] = index
            samples.append(sample)
            print(f'{run}: prompt {index + 1}/{len(prompts)}', flush=True)
        write_json(output / 'samples.json', samples)
    write_json(output / 'comparison.json', dict(results=rows, sampling=settings, prompts=prompts))
    lines = ['# So sánh GPT: cùng dữ liệu và ngân sách token', '',
             '| Tham số | Token train | Val NLL | Test NLL | Test PPL | Thời gian train (s) |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['parameters']:,} | {row['training_tokens']:,} | "
                     f"{row['validation_loss']:.4f} | {row['test_loss']:.4f} | "
                     f"{row['test_perplexity']:.3f} | {row['training_seconds']:.2f} |")
    lines += ['', 'PPL thấp hơn là tốt hơn. Chọn checkpoint bằng validation; test chỉ dùng báo cáo.',
              'Cùng tokenizer, context 256, seed 42, ngân sách 20M target token; chỉ đổi kiến trúc.',
              'Đây là so sánh một seed, hai kích thước, không chứng minh quy luật scaling.',
              'Thời gian của hai lượt chạy khác ngày, có thể chịu ảnh hưởng điều kiện phần cứng.',
              '', '## Toàn bộ mẫu cố định', '',
              'Seed 42, temperature 0.8, top-k 40, top-p 0.9, tối đa 80 token mới.',
              'Giữ nguyên mọi đầu ra; đây chưa phải đánh giá chấm người.', '']
    for index, prompt in enumerate(prompts):
        lines += [f'### {index + 1}. {prompt}', '']
        for row in rows:
            sample = next(s for s in samples if Path(s['run']) == Path(row['run']) and s['prompt_id'] == index)
            lines += [f"**{row['parameters']:,} tham số**", '', sample['prompt'] + sample['continuation'], '']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
