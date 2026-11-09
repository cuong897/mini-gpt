import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from minigpt import demo_web as web


def available_info(preset, model):
    return {'available': True, 'id': model, 'kind':'pilot', 'data_kind':'real',
            'data_fingerprint':'test-data', 'checkpoint_sha256':'test-checkpoint'}


@pytest.mark.parametrize('change', [
    {'prompt': '  '}, {'prompt': 'a' * 4001}, {'models': ['gpt', 'gpt']},
    {'models': ['unknown']}, {'models': [{}]}, {'preset': '../runs'},
    {'max_new_tokens': True}, {'max_new_tokens': 201}, {'seed': -1}, {'seed': 2**32},
    {'temperature': float('nan')}, {'temperature': float('inf')}, {'top_k': 1.5},
    {'top_p': 0}, {'checkpoint': 'other.pt'},
])
def test_demo_rejects_invalid_request_before_loading_models(change):
    with pytest.raises(ValueError):
        web.validate_request({'prompt': 'Tiếng Việt', **change})


def test_routes_each_model_to_its_own_tokenizer_and_keeps_partial_results(monkeypatch):
    calls = []
    monkeypatch.setattr(web, 'model_info', available_info)
    def fake_generate(data, run, prompt, **settings):
        calls.append((data, run, prompt, settings))
        if run.name == 'lstm':
            raise ValueError('checkpoint test unavailable')
        return {'continuation': ' nội dung thử', 'model': run.name,
                **{key:value for key,value in available_info('latest',run.name).items() if key not in ('id','available')}}
    monkeypatch.setattr(web, 'generate', fake_generate)
    result = web.Demo('cpu').run({'prompt': 'Việt Nam', 'models': ['ngram', 'lstm', 'gpt']})
    assert [c[0].name for c in calls] == ['wiki_reviewed_v1', 'wiki_reviewed_v1', 'wiki_main_20261102_v2']
    assert all(c[2] == 'Việt Nam' and c[3]['seed'] == 42 for c in calls)
    assert result['results'][1]['error'] == 'checkpoint test unavailable'
    assert result['results'][2]['result']['model'] == 'gpt'
    assert web.model_paths('pilot', 'gpt')[0].name == 'wiki_reviewed_v1'
    assert web.model_paths('latest', 'gpt')[1].parent.name == 'continuation_20261105'


@contextmanager
def serving(demo):
    server = ThreadingHTTPServer(('127.0.0.1', 0), web.make_handler(demo))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(url, payload, **headers):
    request = Request(url + '/api/generate', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json', **headers})
    return urlopen(request, timeout=10)


def test_http_limits_static_paths_origins_and_input():
    with serving(web.Demo('cpu')) as url:
        with urlopen(url) as response:
            assert 'Mini-GPT'.encode() in response.read()
            assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
        with pytest.raises(HTTPError) as error:
            urlopen(url + '/../../README.md')
        assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            post(url, {'prompt': 'a'}, Origin='https://example.com')
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post(url, {'prompt': 'a'}, Host='another-host:7860')
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post(url, {'prompt': 'a', 'max_new_tokens': 500})
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post(url, {'prompt': 'a' * 40000})
        assert error.value.code == 400


def test_parallel_http_request_cannot_change_generation_rng(monkeypatch):
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(web, 'model_info', available_info)
    def fake_generate(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return {'continuation': '<script>plain text</script>', 'model':'gpt',
                **{key:value for key,value in available_info('latest','gpt').items() if key not in ('id','available')}}
    monkeypatch.setattr(web, 'generate', fake_generate)
    with serving(web.Demo('cpu')) as url:
        answers = []
        def first():
            with post(url, {'prompt': 'a', 'models': ['gpt']}) as response:
                answers.append(json.load(response))
        worker = threading.Thread(target=first)
        worker.start()
        try:
            assert started.wait(3)
            with pytest.raises(HTTPError) as error:
                post(url, {'prompt': 'b', 'models': ['gpt']})
            assert error.value.code == 409
            # Read-only UI requests remain responsive during inference.
            assert urlopen(url + '/style.css', timeout=2).status == 200
        finally:
            release.set()
            worker.join(timeout=5)
        assert len(answers) == 1
        assert answers[0]['results'][0]['result']['continuation'].startswith('<script>')
