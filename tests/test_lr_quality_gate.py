import copy

import pytest

from scripts.report_lr_control import assess_arm, sample_key


def case():
    samples=[dict(arm='candidate',prompt_id=i,temperature=t,seed=s,repeated_4gram_fraction=.02)
             for i in range(11) for t,s in [(0.,42),(.8,42),(.8,7)]]
    reviews=[dict(key=sample_key(s),on_topic=2,coherence=2,grammar=1,reason='Readable and connected') for s in samples]
    return samples,reviews


GATE=dict(default_acceptable_at_least=8,all_acceptable_at_least=17)


def test_lower_loss_cannot_promote_incoherent_text():
    samples,reviews=case()
    for row in reviews: row['coherence']=0
    result=assess_arm({'selected_validation_nll':3.9},{'selected_validation_nll':4.1},samples,reviews,samples,GATE)
    assert result['checks']['validation_improved']
    assert not result['eligible_for_demo'] and result['acceptable_all']==0


def test_all_quality_gates_are_required():
    samples,reviews=case()
    result=assess_arm({'selected_validation_nll':3.9},{'selected_validation_nll':4.1},samples,reviews,samples,GATE)
    assert result['eligible_for_demo']
    repeated=copy.deepcopy(samples)
    for row in repeated: row['repeated_4gram_fraction']=.5
    result=assess_arm({'selected_validation_nll':3.9},{'selected_validation_nll':4.1},repeated,reviews,samples,GATE)
    assert not result['eligible_for_demo']
    with pytest.raises(ValueError,match='33'):
        assess_arm({'selected_validation_nll':3.9},{'selected_validation_nll':4.1},samples[:-1],reviews,samples,GATE)
