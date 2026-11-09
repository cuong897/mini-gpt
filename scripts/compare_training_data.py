"""Paired evaluation for changed train data with byte-identical heldout/tokenizer."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from minigpt.common import read_json, write_json, sha256
from minigpt.experiments import checked_result
from minigpt.generation import generate


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--original-data',required=True)
    parser.add_argument('--clean-data',required=True)
    parser.add_argument('--original-run',required=True)
    parser.add_argument('--clean-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    for filename in ['tokenizer.json']+[f'{s}.{x}' for s in ['validation','test'] for x in ['jsonl','bin','offsets.npy']]:
        assert sha256(Path(args.original_data)/filename)==sha256(Path(args.clean_data)/filename), filename
    assert read_json(Path(args.original_run)/'config.json')==read_json(Path(args.clean_run)/'config.json')
    results,samples=[],[]
    prompts=read_json('configs/prompts.json')
    for label,data,run in [('Original train',args.original_data,args.original_run),('Clean prose train',args.clean_data,args.clean_run)]:
        ev,meta=checked_result(run)
        assert meta['budget_completed'] and meta['tokens_seen']==20000000 and meta['checkpoint_binding_verified']
        manifest=read_json(Path(data)/'manifest.json')
        assert ev['data_fingerprint']==sha256(Path(data)/'manifest.json')
        results.append(dict(label=label,run=run,parameters=meta['parameters'],validation_loss=meta['best_validation_loss'],
                            test_loss=ev['loss'],test_perplexity=ev['perplexity'],training_tokens=meta['tokens_seen'],
                            unique_train_tokens=manifest['splits']['train']['target_tokens'],
                            training_seconds=meta['elapsed_train_seconds'],checkpoint_sha256=ev['checkpoint_sha256']))
        for temperature in [.8,1.0]:
            for i,prompt in enumerate(prompts):
                sample=generate(data,run,prompt,max_new_tokens=80,seed=42,temperature=temperature,top_k=40,top_p=.9,device_name='cuda')
                assert sample['checkpoint_sha256']==ev['checkpoint_sha256']
                ids=sample['generated_ids']
                fours=list(zip(ids,ids[1:],ids[2:],ids[3:]))
                sample.update(label=label,prompt_id=i,repeated_4gram_fraction=sum(n-1 for n in Counter(fours).values())/len(fours) if fours else 0)
                samples.append(sample)
                write_json(output/'samples.json',samples)
                print(f'{label}, T={temperature}: {i+1}/10',flush=True)
    assert results[0]['parameters']==results[1]['parameters']
    metrics=[]
    for label in [r['label'] for r in results]:
        for temperature in [.8,1.0]:
            subset=[s for s in samples if s['label']==label and s['sampling']['temperature']==temperature]
            metrics.append(dict(label=label,temperature=temperature,samples=len(subset),
                                mean_repeated_4gram_fraction=sum(s['repeated_4gram_fraction'] for s in subset)/len(subset),
                                eos_count=sum(s['stop_reason']=='eos' for s in subset)))
    write_json(output/'comparison.json',dict(results=results,sampling_metrics=metrics))
    lines=['# Đối chiếu tập train gốc và văn xuôi đã lọc','',
           '| Tập train | Token riêng trong tập | Token đã học | Val NLL | Test NLL | Test PPL |',
           '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['label']} | {r['unique_train_tokens']:,} | {r['training_tokens']:,} | {r['validation_loss']:.4f} | {r['test_loss']:.4f} | {r['test_perplexity']:.3f} |")
    lines+=['','Cùng tokenizer và validation/test byte-identical; cùng cấu hình/seed29,4M và ngân sách20M, khởi tạo ngẫu nhiên.',
            'Bộ train sạch nhỏ hơn nên số lượt đọc tương đương lớn hơn. Kết quả phản ánh toàn bộ bộ lọc/giảm corpus, không tách riêng từng quy tắc.',
            'Heldout giữ nguyên cả nhiễu để so được; chưa chứng nhận sạch ngữ nghĩa. Một seed, chưa chấm người độc lập.','',
            '## Chỉ số lặp','', '| Train | T | Tỷ lệ 4-gram BPE lặp TB | Dừng EOS /10 |','|---|---:|---:|---:|']
    for m in metrics:
        lines.append(f"| {m['label']} | {m['temperature']} | {m['mean_repeated_4gram_fraction']:.3f} | {m['eos_count']} |")
    lines+=['','Lặp thấp không bảo đảm đúng hoặc mạch lạc. Tất cả mẫu dùng seed42/k40/p0,9/80token.','', '## Toàn bộ40mẫu','']
    for i,prompt in enumerate(prompts):
        lines += [f'### {i+1}. {prompt}','']
        for s in (s for s in samples if s['prompt_id']==i):
            lines += [f"**{s['label']} · T={s['sampling']['temperature']}**",'',s['prompt']+s['continuation'],'']
    (output/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(results=results,sampling_metrics=metrics),indent=2))


if __name__=='__main__':
    main()
