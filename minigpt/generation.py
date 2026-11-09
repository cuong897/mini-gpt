import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .common import BOS, EOS, PAD, UNK, read_json, seed_all, sha256, write_json
from .data import encode_text, verify_artifacts
from .engine import device_for, load_neural
from .ngram import NGram


def validate_sampling(temperature, top_k, top_p):
    if not math.isfinite(temperature) or temperature < 0 or not isinstance(top_k, int) or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature>=0 hữu hạn; top_k nguyên >=0; 0<top_p<=1")


def sample_token(logits, generator, temperature=1., top_k=0, top_p=1., greedy=False):
    validate_sampling(temperature, top_k, top_p)
    logits = logits.double().clone()
    logits[[PAD, BOS, UNK]] = -float("inf")
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
        raise ValueError("Logits không có phân phối hợp lệ")
    if greedy or temperature == 0:
        return int(logits.argmax())
    # Center first: a valid tiny positive temperature must not create +inf.
    logits = (logits - logits.max()) / temperature
    if top_k:
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        filtered = torch.full_like(logits, -float("inf"))
        filtered[indices] = values
        logits = filtered
    if top_p < 1:
        sorted_logits, order = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Include the token crossing the threshold and always keep the top one.
        remove = (probs.cumsum(-1) - probs) >= top_p
        logits[order[remove]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("Không còn token hợp lệ sau lọc sampling")
    return int(torch.multinomial(probs.cpu(), 1, generator=generator))


@torch.no_grad()
def generate(data, run, prompt, max_new_tokens=100, seed=42, temperature=1., top_k=0, top_p=1.,
             greedy=False, device_name="auto", checkpoint="best.pt"):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens phải >=0")
    validate_sampling(temperature, top_k, top_p)
    seed_all(seed)
    run = Path(run)
    fingerprint = verify_artifacts(data)
    tokenizer = Tokenizer.from_file(str(Path(data) / "tokenizer.json"))
    ids = [BOS] + encode_text(tokenizer, prompt)
    neural = not (run / "ngram.json").exists()
    checkpoint_path = run / (checkpoint if neural else "ngram.sqlite")
    digest_before = sha256(checkpoint_path)
    if neural:
        device = device_for(device_name)
        model, state = load_neural(run / checkpoint, device)
        context, name = model.context, state["config"]["model"]
    else:
        state = read_json(run / "ngram.json")
        if sha256(run / "ngram.sqlite") != state["counts_sha256"]:
            raise ValueError("N-gram counts đã thay đổi")
        model = NGram(run / "ngram.sqlite", state["vocab_size"], state["order"], state["alpha"], state["delta"])
        context, name = state["context"], "ngram"
    if state["data_fingerprint"] != fingerprint:
        if not neural:
            model.close()
        raise ValueError("Tokenizer/dữ liệu không khớp checkpoint")
    settings = dict(seed=seed, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy, max_new_tokens=max_new_tokens)
    generator = torch.Generator().manual_seed(seed)
    continuation, stopped = [], "length"
    try:
        for _ in range(max_new_tokens):
            history = ids[-context:]
            if neural:
                logits = model(torch.tensor([history], device=device))[:, -1, :].squeeze(0).cpu()
            else:
                p = torch.from_numpy(model.distribution(history))
                logits = p.log()
            token = sample_token(logits, generator, temperature, top_k, top_p, greedy)
            if token == EOS:
                stopped = "eos"
                break
            ids.append(token)
            continuation.append(token)
    finally:
        if not neural:
            model.close()
    if sha256(checkpoint_path) != digest_before:
        raise ValueError("Checkpoint thay đổi trong khi sinh; không gắn provenance của file mới cho model cũ")
    return {"model": name, "prompt": prompt, "continuation": tokenizer.decode(continuation),
            "generated_ids": continuation, "stop_reason": stopped, "sampling": settings,
            "data_fingerprint": fingerprint, "checkpoint": checkpoint if neural else "ngram.sqlite",
            "checkpoint_sha256": digest_before, "run": str(run),
            "kind": state.get("kind", state.get("config", {}).get("kind")), "data_kind": state["data_kind"],
            "context_policy": "last context tokens, reset/recompute; generation differs from disjoint-block evaluation"}
