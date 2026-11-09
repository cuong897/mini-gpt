"""Post-diagnostic control: change only top_p from 0.9 to 1.0, all 11 prompts."""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import torch
from minigpt.common import read_json, sha256, write_json
from minigpt.generation import generate
from scripts.diagnose_conditioning import OUT, SOURCE, LIVE_RUN


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    protocol=read_json(OUT/'protocol.json')
    write_json(OUT/'sampling_control_protocol.json',dict(
        timing='Exploratory after finding that top_p excludes the token beginning quốc gia',
        purpose='Check whether removing only nucleus truncation fixes observed text failures',
        changed_parameter=dict(top_p=[.9,1.]), fixed=dict(temperature=.8,top_k=40,seed=42,max_new_tokens=80),
        prompts=protocol['prompts'],checkpoint_sha256=protocol['live_checkpoint_sha256'],
        script_sha256=sha256(__file__),scope='No training, checkpoint reselection, test evaluation, or deployment'))
    samples=[]
    for prompt in protocol['prompts']:
        row=generate(SOURCE,LIVE_RUN,prompt,max_new_tokens=80,seed=42,
                     temperature=.8,top_k=40,top_p=1.,device_name='cuda')
        assert row['checkpoint_sha256']==protocol['live_checkpoint_sha256']
        samples.append(row)
        write_json(OUT/'sampling_control_samples.json',samples)
        print(prompt+': '+row['continuation'],flush=True)
    baseline={s['prompt']:s for s in read_json(OUT/'live_samples.json') if s['temperature']==.8 and s['seed']==42}
    lines=['# Đối chứng top-p', '', 'Chỉ đổi top_p 0,9 → 1,0; giữ T0,8, k40, seed42, checkpoint và 80 token. Thăm dò sau chẩn đoán, không chọn ví dụ đẹp.', '']
    for row in samples:
        lines += [f"## {row['prompt']}",'','**p=0,9**','',row['prompt']+baseline[row['prompt']]['continuation'],
                  '', '**p=1,0**','',row['prompt']+row['continuation'],'']
    (OUT/'sampling_control.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
