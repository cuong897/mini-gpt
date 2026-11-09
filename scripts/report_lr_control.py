"""Aggregate fixed-budget LR results and explicit per-sample qualitative reviews."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.common import read_json, sha256, write_json
from scripts.lr_control_20261105 import OUT, ARMS, PARENT


def sample_key(row):
    return f"{row['arm']}:{row['prompt_id']}:{row['temperature']}:{row['seed']}"


def assess_arm(metric, baseline_metric, samples, reviews, baseline_samples, gate):
    expected = {(i,t,s) for i in range(11) for t,s in [(0.,42),(.8,42),(.8,7)]}
    actual = {(r['prompt_id'],r['temperature'],r['seed']) for r in samples}
    if len(samples)!=33 or actual!=expected:
        raise ValueError('Need all 33 fixed samples exactly once')
    review_map = {r['key']:r for r in reviews}
    if len(review_map)!=len(reviews):
        raise ValueError('Duplicate review keys')
    if any(sample_key(s) not in review_map for s in samples):
        raise ValueError('Missing a qualitative review')
    acceptable=[]
    for s in samples:
        r=review_map[sample_key(s)]
        if any(type(r[k]) is not int or r[k] not in [0,1,2] for k in ['on_topic','coherence','grammar']) or not r.get('reason'):
            raise ValueError('Each review needs bounded scores and a written reason')
        if r['on_topic']==2 and r['coherence']==2 and r['grammar']>=1:
            acceptable.append(s)
    default=lambda rows:[r for r in rows if r['temperature']==.8 and r['seed']==42]
    d=default(samples)
    b=default(baseline_samples)
    assert len(d)==len(b)==11
    repeat=sum(r['repeated_4gram_fraction'] for r in d)/11
    baseline_repeat=sum(r['repeated_4gram_fraction'] for r in b)/11
    checks=dict(validation_improved=metric['selected_validation_nll']<baseline_metric['selected_validation_nll'],
                default_quality=len(default(acceptable))>=gate['default_acceptable_at_least'],
                overall_quality=len(acceptable)>=gate['all_acceptable_at_least'],
                repetition=repeat<=baseline_repeat+.01)
    return dict(acceptable_all=len(acceptable),acceptable_default=len(default(acceptable)),
                mean_default_repeated_4gram=repeat,checks=checks,eligible_for_demo=all(checks.values()))


def main():
    protocol=read_json(OUT/'protocol.json')
    metrics=read_json(OUT/'metrics.json')
    samples=read_json(OUT/'samples.json')
    review=read_json(OUT/'qualitative_review.json')
    assert len(samples)==len(review['rows'])==99
    assert {sample_key(s) for s in samples}=={r['key'] for r in review['rows']}
    baseline=next(r for r in metrics if r['arm']=='parent')
    base_samples=[r for r in samples if r['arm']=='parent']
    results=[]
    for metric in metrics:
        name=metric['arm']
        run=PARENT if name=='parent' else Path(ARMS[name][0])
        summary=read_json(run/'summary.json')
        current=[r for r in samples if r['arm']==name]
        quality=assess_arm(metric,baseline,current,review['rows'],base_samples,protocol['deployment_gate'])
        results.append(dict(**metric,**quality,completed_cumulative_training_tokens=summary['cumulative_training_tokens'],
                            last_validation_nll=summary['validation_last']['loss'],stage_seconds=summary['elapsed_train_seconds']))
    write_json(OUT/'comparison.json',dict(results=results,reviewer=review['reviewer'],
              validation_winner=min(results,key=lambda r:r['selected_validation_nll'])['arm'],
              eligible_for_demo=[r['arm'] for r in results if r['eligible_for_demo']]))
    lines=['# Đối chứng learning rate: số đo và đánh giá văn bản', '',
           '| Nhánh | Tổng token hoàn tất | Validation NLL tốt nhất | NLL cuối | Đoạn đạt /33 | Mặc định đạt /11 | Lặp 4-gram mặc định | Đủ điều kiện đổi demo |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['arm']} | {r['completed_cumulative_training_tokens']:,} | {r['selected_validation_nll']:.5f} | {r['last_validation_nll']:.5f} | {r['acceptable_all']} | {r['acceptable_default']} | {r['mean_default_repeated_4gram']:.3f} | {r['eligible_for_demo']} |")
    lines+=['','Điểm chất lượng do assistant chấm toàn bộ 99 mẫu theo rubric đặt trước; chưa có người chấm độc lập. Chỉ số lặp không thay thế đánh giá ý nghĩa.', '', '## Ghi nhận từng mẫu', '']
    ratings={r['key']:r for r in review['rows']}
    for s in samples:
        r=ratings[sample_key(s)]
        lines += [f"### {sample_key(s)} · {s['prompt']}", '',s['prompt']+s['continuation'],'',
                  f"Bám chủ đề {r['on_topic']}/2; mạch lạc {r['coherence']}/2; ngữ pháp {r['grammar']}/2. {r['reason']}",'']
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,(run,lr) in ARMS.items():
        rows=[json.loads(l) for l in (Path(run)/'metrics.jsonl').read_text().splitlines()]
        vals=[r for r in rows if 'validation_loss' in r]
        axes[0].plot([0]+[r['tokens_seen']/1e6 for r in vals],
                     [baseline['selected_validation_nll']]+[r['validation_loss'] for r in vals],'o-',label=f'Peak LR {lr:g}')
        axes[1].plot([r['tokens_seen']/1e6 for r in rows],[r['learning_rate'] for r in rows],label=f'Peak LR {lr:g}')
    axes[0].set(xlabel='Additional target tokens (millions)',ylabel='Validation NLL',title='Same source checkpoint, data and token budget')
    axes[1].set(xlabel='Additional target tokens (millions)',ylabel='Learning rate',title='Same warmup and cosine schedule')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'validation_lr.png',dpi=160)
    plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
