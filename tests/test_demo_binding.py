"""Reject mislabeled checkpoint copies even when corpus fingerprints match."""
import pytest
import torch

from minigpt import demo_web as web
from minigpt.common import PROTOCOL, sha256, write_json
from minigpt.models import build_model, parameter_count


def checkpoint_case(tmp_path, monkeypatch, actual='gpt', label='gpt'):
    data, run = tmp_path / 'data', tmp_path / 'run'
    data.mkdir(); run.mkdir()
    write_json(data / 'manifest.json', {'kind':'synthetic','vocab_size':32,
                                      'splits':{'train':{'documents':3}}})
    (data / 'tokenizer.json').write_text('{}',encoding='utf-8')
    architecture = ({'context':8,'dim':8,'heads':2,'layers':1,'dropout':0} if actual=='gpt'
                    else {'context':8,'embedding_dim':8,'hidden_dim':8,'layers':1,'dropout':0})
    config = {'model':actual,'kind':'smoke','architecture':architecture}
    model = build_model(config,32)
    state = {'config':config,'model_state':model.state_dict(),'vocab_size':32,
             'parameters':parameter_count(model),'data_fingerprint':sha256(data/'manifest.json'),
             'data_kind':'synthetic','protocol':PROTOCOL,'tokens_seen':6}
    torch.save(state,run/'best.pt')
    write_json(run/'summary.json', {'model':label,'kind':'smoke','tokens_seen':20,
                                  'parameters':parameter_count(model),
                                  'data_fingerprint':state['data_fingerprint']})
    monkeypatch.setattr(web,'ROOT',tmp_path)
    monkeypatch.setattr(web,'model_paths',lambda *args:(data,run))
    return data,run,state


def test_wrong_model_checkpoint_with_same_fingerprint_is_not_advertised(tmp_path,monkeypatch):
    checkpoint_case(tmp_path,monkeypatch,actual='lstm',label='gpt')
    info=web.model_info('latest','gpt')
    assert info['available'] is False


def test_card_uses_actual_best_checkpoint_training_budget(tmp_path,monkeypatch):
    _,run,_=checkpoint_case(tmp_path,monkeypatch)
    info=web.model_info('latest','gpt')
    assert info['available'] is True
    assert info['training_tokens']==6  # Summary is20, but best weights have seen6.
    assert info['checkpoint_sha256']==sha256(run/'best.pt')


def test_continued_card_counts_parent_plus_actual_best_stage_tokens(tmp_path,monkeypatch):
    _,run,state=checkpoint_case(tmp_path,monkeypatch)
    parent=tmp_path/'parent'
    parent.mkdir()
    torch.save(state,parent/'last.pt')
    write_json(run/'lineage.json', {'source_checkpoint':str(parent/'last.pt'),
                                   'source_sha256':sha256(parent/'last.pt'),
                                   'parent_stage_tokens':6,'prior_training_tokens':6})
    write_json(run/'summary.json', {'model':'gpt','kind':'smoke','tokens_seen':20,
                                   'parameters':state['parameters'],
                                   'data_fingerprint':state['data_fingerprint'],
                                   'prior_training_tokens':6,'cumulative_training_tokens':26})
    info=web.model_info('latest','gpt')
    assert info['available']
    assert info['training_tokens']==12  # Actual best=6, prior=6; final summary=26.
    # An altered ancestor must not silently retain the cumulative label.
    (parent/'last.pt').write_bytes(b'changed')
    assert not web.model_info('latest','gpt')['available']


def test_continued_card_rejects_incorrect_prior_budget(tmp_path,monkeypatch):
    _,run,state=checkpoint_case(tmp_path,monkeypatch)
    parent=tmp_path/'parent'
    parent.mkdir()
    torch.save(state,parent/'last.pt')
    write_json(run/'lineage.json', {'source_checkpoint':str(parent/'last.pt'),
                                   'source_sha256':sha256(parent/'last.pt'),
                                   'parent_stage_tokens':6,'prior_training_tokens':60})
    assert not web.model_info('latest','gpt')['available']


def test_corrupt_checkpoint_is_reported_unavailable(tmp_path,monkeypatch):
    _,run,_=checkpoint_case(tmp_path,monkeypatch)
    (run/'best.pt').write_bytes(b'broken checkpoint')
    assert web.model_info('latest','gpt')['available'] is False


@pytest.mark.parametrize('replacement', [{'model':'lstm'}, {'checkpoint_sha256':'different-file'},
                                          {'data_fingerprint':'different-data'}])
def test_generation_identity_change_isolated_as_model_error(tmp_path,monkeypatch,replacement):
    _,run,state=checkpoint_case(tmp_path,monkeypatch)
    result={'model':'gpt','kind':'smoke','data_kind':'synthetic',
            'data_fingerprint':state['data_fingerprint'],
            'checkpoint_sha256':sha256(run/'best.pt'),**replacement}
    monkeypatch.setattr(web,'generate',lambda *args,**kwargs:result)
    answer=web.Demo('cpu').run({'prompt':'test','models':['gpt']})
    assert 'error' in answer['results'][0]
    assert 'result' not in answer['results'][0]
