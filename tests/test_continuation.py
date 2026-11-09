import copy

import pytest
import torch

from minigpt.__main__ import make_fixture
from minigpt.common import read_json, sha256, write_json
from minigpt.data import prepare
from minigpt.engine import train


@pytest.fixture
def stage_case(tmp_path):
    raw, data = tmp_path / 'raw.jsonl', tmp_path / 'data'
    make_fixture(raw)
    prepare(raw, data, vocab_size=280, ratios=(.7, .15, .15), kind='synthetic')
    config = dict(model='gpt', kind='smoke', seed=4, deterministic=True,
                  architecture=dict(context=8, dim=12, heads=2, layers=1, dropout=.1),
                  batch_size=2, accumulation=2, max_steps=3, max_tokens=83,
                  learning_rate=.003, warmup_steps=1, eval_every=1, precision='fp32', cpu_threads=2)
    path, parent = tmp_path / 'config.json', tmp_path / 'parent'
    write_json(path, config)
    train(path, data, parent, 'cpu')
    return data, config, parent


def test_continuation_preserves_state_and_source_but_resets_stage(tmp_path, stage_case):
    data, config, parent = stage_case
    source = parent / 'last.pt'
    digest = sha256(source)
    config.update(learning_rate=.0001, warmup_steps=0, max_tokens=100, max_steps=5)
    path, stage = tmp_path / 'next.json', tmp_path / 'stage'
    write_json(path, config)
    summary = train(path, data, stage, 'cpu', continue_from=source, stop_after=0)
    a = torch.load(source, weights_only=True)
    b = torch.load(stage / 'last.pt', weights_only=True)
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
    for index, state in a['optimizer_state']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b['optimizer_state']['state'][index][key], rtol=0, atol=0)
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    assert b['step'] == b['tokens_seen'] == summary['tokens_seen'] == 0
    assert b['best_snapshot'] is None
    assert summary['cumulative_training_tokens'] == 83
    assert read_json(stage / 'lineage.json')['source_sha256'] == digest == sha256(source)
    # Train and then resume this stage; inherited cost remains correctly recorded.
    summary = train(path, data, stage, 'cpu', resume=stage/'last.pt')
    assert summary['budget_completed']
    assert summary['cumulative_training_tokens'] == 183


def test_continuation_rejects_architecture_change_and_ambiguous_mode(tmp_path, stage_case):
    data, original, parent = stage_case
    config = copy.deepcopy(original)
    config['architecture']['heads'] = 3
    path = tmp_path / 'bad.json'
    write_json(path, config)
    with pytest.raises(ValueError, match='cùng kiến trúc'):
        train(path, data, tmp_path/'bad', 'cpu', continue_from=parent/'last.pt')
    with pytest.raises(ValueError, match='không dùng cả hai'):
        train(path, data, tmp_path/'both', 'cpu', resume=parent/'last.pt', continue_from=parent/'last.pt')
