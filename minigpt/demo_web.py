"""Local browser demo backed by the existing, provenance-checked generator."""
import argparse
import json
import math
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .common import PROTOCOL, read_json, sha256
from .generation import generate
from .models import build_model, parameter_count

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web" / "demo"
LABELS = {"ngram": "N-gram", "lstm": "LSTM", "gpt": "Mini-GPT"}
PRESETS = {"latest": "Tốt nhất theo validation", "pilot": "Cùng dữ liệu pilot"}


def model_paths(preset, model):
    if preset == "latest" and model == "gpt":
        return ROOT / "data/wiki_main_20261102_v2", ROOT / "runs/continuation_20261105/gpt"
    return ROOT / "data/wiki_reviewed_v1", ROOT / "runs/reviewed_pilot" / model


def inherited_training_tokens(run, fingerprint, architecture, visited=None):
    """Bind prior-stage token counts to the actual source checkpoint chain."""
    visited = set() if visited is None else visited
    run = Path(run).resolve()
    if run in visited:
        raise ValueError("Chuỗi checkpoint nguồn bị vòng lặp")
    visited.add(run)
    path = run / "lineage.json"
    if not path.exists():
        return 0
    lineage = read_json(path)
    source = Path(lineage["source_checkpoint"])
    digest = sha256(source)
    if digest != lineage["source_sha256"]:
        raise ValueError("Checkpoint nguồn đã thay đổi")
    import torch
    parent = torch.load(source, map_location="meta", mmap=True, weights_only=True)
    if (parent["data_fingerprint"] != fingerprint or parent["protocol"] != PROTOCOL
            or parent["config"]["model"] != "gpt" or parent["config"]["architecture"] != architecture
            or parent["tokens_seen"] != lineage["parent_stage_tokens"]):
        raise ValueError("Nguồn giai đoạn học không khớp checkpoint")
    total = parent["tokens_seen"] + inherited_training_tokens(source.parent, fingerprint, architecture, visited)
    if total != lineage["prior_training_tokens"] or sha256(source) != digest:
        raise ValueError("Tổng token giai đoạn trước không khớp nguồn")
    return total


def model_info(preset, model):
    data, run = model_paths(preset, model)
    info = {"id": model, "label": LABELS[model], "available": False}
    try:
        manifest = read_json(data / "manifest.json")
        metadata = read_json(run / ("ngram.json" if model == "ngram" else "summary.json"))
        checkpoint = run / ("ngram.sqlite" if model == "ngram" else "best.pt")
        if not checkpoint.is_file() or not (data / "tokenizer.json").is_file():
            raise ValueError("Thiếu checkpoint hoặc tokenizer")
        if sha256(data / "manifest.json") != metadata["data_fingerprint"]:
            raise ValueError("Dữ liệu không khớp checkpoint đã ghi nhận")
        if metadata["model"] != model:
            raise ValueError("Nhãn mô hình không khớp bản đã lưu")
        digest = sha256(checkpoint)
        if model == "ngram":
            state = metadata
            parameters = None
            training_tokens = state["train_unique_target_tokens"]
            if state["counts_sha256"] != digest:
                raise ValueError("Bản đếm n-gram đã thay đổi")
        else:
            import torch
            state = torch.load(checkpoint, map_location="meta", mmap=True, weights_only=True)
            if state["config"]["model"] != model or state["config"]["kind"] != metadata["kind"]:
                raise ValueError("Loại checkpoint thật không khớp nhãn mô hình")
            with torch.device("meta"):
                actual_model = build_model(state["config"], state["vocab_size"])
                actual_model.load_state_dict(state["model_state"], strict=True)
            parameters = parameter_count(actual_model)
            if state["parameters"] != parameters or metadata["parameters"] != parameters:
                raise ValueError("Số tham số không khớp checkpoint thật")
            # A best checkpoint may precede the end of a run: label the weights
            # being demonstrated, rather than the later summary's token budget.
            stage_training_tokens = state["tokens_seen"]
            prior_tokens = inherited_training_tokens(run, state["data_fingerprint"], state["config"]["architecture"]) if model == "gpt" else 0
            if (metadata.get("prior_training_tokens", 0) != prior_tokens
                    or (prior_tokens and metadata.get("cumulative_training_tokens") != prior_tokens + metadata["tokens_seen"])):
                raise ValueError("Ngân sách học cộng dồn không khớp nguồn")
            training_tokens = prior_tokens + stage_training_tokens
        if (state["data_fingerprint"] != metadata["data_fingerprint"]
                or state["vocab_size"] != manifest["vocab_size"]
                or state["data_kind"] != manifest["kind"] or state["protocol"] != PROTOCOL):
            raise ValueError("Checkpoint không khớp dữ liệu/tokenizer/giao thức")
        if sha256(checkpoint) != digest:
            raise ValueError("Checkpoint thay đổi khi kiểm tra nhãn")
        info.update(available=True, kind=metadata["kind"], data_kind=state["data_kind"],
                    documents=sum(s["documents"] for s in manifest["splits"].values()),
                    training_tokens=training_tokens,
                    vocabulary=manifest["vocab_size"], parameters=parameters,
                    data_fingerprint=metadata["data_fingerprint"],
                    checkpoint_sha256=digest,
                    run=run.relative_to(ROOT).as_posix())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        info["reason"] = f"Chưa sẵn sàng: {error}"
    return info


def validate_request(payload):
    allowed = {"preset", "models", "prompt", "max_new_tokens", "seed", "temperature", "top_k", "top_p"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Yêu cầu không hợp lệ")
    preset = payload.get("preset", "latest")
    selected = payload.get("models", list(LABELS))
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("Bộ mô hình không hợp lệ")
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 3
            or any(not isinstance(m, str) or m not in LABELS for m in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Chọn từ 1 đến 3 mô hình khác nhau")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Nhập đoạn mở đầu từ 1 đến 4.000 ký tự")
    settings = {"max_new_tokens": payload.get("max_new_tokens", 80), "seed": payload.get("seed", 42),
                "temperature": payload.get("temperature", .8), "top_k": payload.get("top_k", 40),
                "top_p": payload.get("top_p", .9)}
    for key, lo, hi in [("max_new_tokens", 1, 200), ("seed", 0, 2**32 - 1), ("top_k", 0, 8000)]:
        if type(settings[key]) is not int or not lo <= settings[key] <= hi:
            raise ValueError(f"{key} phải là số nguyên từ {lo} đến {hi}")
    for key, lo, hi in [("temperature", 0, 2), ("top_p", 0, 1)]:
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key} phải nằm trong khoảng {lo} đến {hi}")
    if settings["top_p"] == 0:
        raise ValueError("Top-p phải lớn hơn 0")
    return preset, selected, prompt, settings


class Demo:
    def __init__(self, device="auto"):
        self.device = device
        self.lock = threading.Lock()

    def catalog(self):
        return {"default_preset": "latest", "presets": [
            {"id": preset, "label": label, "models": [model_info(preset, m) for m in LABELS]}
            for preset, label in PRESETS.items()]}

    def run(self, payload):
        preset, selected, prompt, settings = validate_request(payload)
        # generate() seeds global RNGs. Serialize complete requests, including
        # all models, so simultaneous browser tabs cannot change each other's RNG.
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Đang sinh văn bản ở một cửa sổ khác. Hãy thử lại sau.")
        try:
            results = []
            for model in selected:
                info = model_info(preset, model)
                if not info["available"]:
                    results.append({"id": model, "info": info, "error": info["reason"]})
                    continue
                data, run = model_paths(preset, model)
                started = time.perf_counter()
                try:
                    result = generate(data, run, prompt, device_name=self.device, **settings)
                    expected = {"model": model, **{key: info[key] for key in
                                ("kind", "data_kind", "data_fingerprint", "checkpoint_sha256")}}
                    if any(result.get(key) != value for key, value in expected.items()):
                        raise ValueError("Kết quả không khớp loại mô hình hoặc checkpoint đã kiểm tra; hãy tải lại danh sách")
                    results.append({"id": model, "info": info, "result": result,
                                    "seconds": time.perf_counter() - started})
                except Exception as error:
                    # One missing/bad model should not hide other model outputs.
                    results.append({"id": model, "info": info, "error": str(error)})
            return {"preset": preset, "prompt": prompt, "settings": settings, "results": results}
        finally:
            self.lock.release()


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in hosts or (origin and origin not in {f"http://{h}" for h in hosts}):
                self.reply(403, {"error": "Chỉ chấp nhận truy cập từ demo trên máy này"})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path == "/api/models":
                self.reply(200, demo.catalog())
                return
            static = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                      "/style.css": ("style.css", "text/css")}
            if path not in static:
                self.reply(404, {"error": "Không tìm thấy trang"})
                return
            name, mime = static[path]
            self.reply(200, (ASSETS / name).read_bytes(), mime + "; charset=utf-8")

        def do_POST(self):
            if not self.local_request():
                return
            if urlsplit(self.path).path != "/api/generate":
                self.reply(404, {"error": "Không tìm thấy chức năng"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Yêu cầu phải dùng JSON")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("Yêu cầu quá dài hoặc rỗng")
                payload = json.loads(self.rfile.read(length))
                result = demo.run(payload)
            except (ValueError, UnicodeError) as error:
                self.reply(400, {"error": str(error)})
            except BlockingIOError as error:
                self.reply(409, {"error": str(error)})
            except Exception:
                self.reply(500, {"error": "Không thể sinh văn bản. Kiểm tra cửa sổ chạy demo."})
            else:
                self.reply(200, result)

    return Handler


def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demo 2 baseline và mini-GPT trên trình duyệt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port phải từ 1 đến 65535")
    import torch
    torch.set_num_threads(2)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Demo(args.device)))
    server.daemon_threads = True
    print(f"Demo: http://127.0.0.1:{args.port} (Ctrl+C để dừng)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
