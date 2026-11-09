"""CPU-only numerical lessons; never read or alter research checkpoints/data."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minigpt.common import read_json, write_json
from minigpt.models import GPT, build_model, parameter_count


def loss_and_gradient():
    # Treat three logits as the trainable toy parameters, not as a full LM.
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1])
    optimizer = torch.optim.SGD([logits], lr=.3)
    before = logits.softmax(-1).detach().tolist()
    loss = nn.functional.cross_entropy(logits[None], target)
    loss_before = loss.item()
    optimizer.zero_grad()
    loss.backward()
    gradient = logits.grad.detach().tolist()
    torch.testing.assert_close(logits.grad, torch.tensor([1/3, -2/3, 1/3], dtype=torch.float64))
    optimizer.step()
    after = logits.softmax(-1).detach().tolist()
    loss_after = nn.functional.cross_entropy(logits[None], target).item()
    assert after[1] > before[1] and loss_after < loss_before
    torch.testing.assert_close(logits.detach(), torch.tensor([-.1, .2, -.1], dtype=torch.float64))
    return {'probabilities_before':before, 'loss_before':loss_before,
            'gradient':gradient, 'logits_after':logits.detach().tolist(),
            'probabilities_after':after, 'loss_after':loss_after}


def attention_example():
    query = torch.tensor([1., 0.], dtype=torch.float64)
    keys = torch.eye(2, dtype=torch.float64)
    values = torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64)
    scores = query @ keys.T / math.sqrt(2)
    weights = scores.softmax(-1)
    output = weights @ values
    assert math.isclose(weights.sum().item(), 1.)
    assert math.isclose(output[0].item(), 2 * weights[0].item())
    assert math.isclose(output[1].item(), 4 * weights[1].item())
    return {'scores':scores.tolist(), 'weights':weights.tolist(), 'output':output.tolist()}


def weighted_perplexity():
    total_nll, tokens = 2 * 1 + 8 * 3, 2 + 8
    correct_loss = total_nll / tokens
    correct_ppl = math.exp(correct_loss)
    wrong_loss = (1 + 3) / 2
    wrong_ppl = (math.exp(1) + math.exp(3)) / 2
    assert correct_loss == 2.6 and not math.isclose(correct_ppl, wrong_ppl)
    return {'total_nll':total_nll, 'tokens':tokens, 'correct_mean_loss':correct_loss,
            'correct_ppl':correct_ppl, 'wrong_unweighted_loss':wrong_loss,
            'wrong_average_batch_ppl':wrong_ppl}


def lstm_gates():
    # PyTorch LSTM uses gate chunks i,f,g,o. Compare one manual timestep.
    torch.manual_seed(42)
    lstm = nn.LSTM(input_size=3, hidden_size=2, num_layers=1, batch_first=True).double().eval()
    x = torch.tensor([[.1, .2, -.3]], dtype=torch.float64)
    h0 = torch.zeros(1, 2, dtype=torch.float64)
    c0 = torch.zeros_like(h0)
    with torch.no_grad():
        gates = (x @ lstm.weight_ih_l0.T + lstm.bias_ih_l0
                 + h0 @ lstm.weight_hh_l0.T + lstm.bias_hh_l0)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        cell = f * c0 + i * g
        hidden = o * cell.tanh()
        output, (hn, cn) = lstm(x[:, None, :], (h0[None], c0[None]))
        torch.testing.assert_close(hidden, output[:, 0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hidden, hn[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cell, cn[0], rtol=1e-12, atol=1e-12)
    return {'input_gate':i.tolist(), 'forget_gate':f.tolist(), 'candidate':g.tolist(),
            'output_gate':o.tolist(), 'cell':cell.tolist(), 'hidden':hidden.tolist(),
            'manual_matches_pytorch':True}


def gpt_shapes_and_causality():
    torch.manual_seed(42)
    model = GPT(32, context=4, dim=16, heads=2, layers=1, dropout=0).cpu().eval()
    first = torch.tensor([[1, 4, 5, 6]])
    changed_future = torch.tensor([[1, 4, 5, 7]])
    with torch.no_grad():
        left, right = model(first), model(changed_future)
        # PAD/BOS logits are -inf by design. Compare the finite support.
        torch.testing.assert_close(left[:, :3, 2:], right[:, :3, 2:], rtol=1e-7, atol=1e-7)
        assert torch.isfinite(left[:, :, 2:]).all()
    return {'input_shape':list(first.shape), 'logits_shape':list(left.shape),
            'future_changed_at_index':3, 'earlier_logits_unchanged':True}


def main_parameter_count():
    config = read_json(ROOT / 'configs/gpt_main.json')
    with torch.device('meta'):
        actual = parameter_count(build_model(config, 8000))
    a = config['architecture']
    d, c, layers = a['dim'], a['context'], a['layers']
    formula = 8000*d + c*d + layers*(12*d*d + 13*d) + 2*d
    assert a['tie_weights'] is True and formula == actual == 13817856
    return {'vocabulary_assumed':8000, 'formula':formula, 'actual_unique_parameters':actual,
            'model_built_on_meta_without_weight_storage':True}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Bài thực hành số nhỏ để hiểu mini-GPT')
    parser.add_argument('--output', help='Tùy chọn lưu kết quả JSON, không ghi vào run/checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {'scope':'toy educational calculations only; CPU; no research checkpoint/data loaded',
               'loss_gradient':loss_and_gradient(), 'attention':attention_example(),
               'weighted_perplexity':weighted_perplexity(), 'lstm_gates':lstm_gates(),
               'gpt_causality':gpt_shapes_and_causality(), 'main_parameter_count':main_parameter_count()}
    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    print('Đã kiểm tra các ví dụ số. Đây không phải phép đo chất lượng model chính.')


if __name__ == '__main__':
    main()
