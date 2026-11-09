# Mini-GPT tiếng Việt

Huấn luyện mô hình ngôn ngữ tiếng Việt từ đầu bằng PyTorch, với ba mô hình **N-gram, LSTM và GPT decoder-only**. Dự án tập trung vào luồng xử lý dữ liệu, huấn luyện, đánh giá và sinh văn bản trên cùng tokenizer và giao thức đánh giá. Không sử dụng trọng số pretrained.

## Chức năng

- Chuẩn hóa Unicode NFC, loại trùng tài liệu và gom các tài liệu có đoạn dài trùng nhau trước khi chia tập.
- Học tokenizer BPE chỉ trên tập train; giữ ranh giới tài liệu khi tạo context.
- N-gram dùng bộ đếm SQLite; LSTM và GPT được huấn luyện từ khởi tạo ngẫu nhiên.
- Theo dõi loss, perplexity, ngân sách target token, checkpoint và fingerprint dữ liệu.
- Khôi phục huấn luyện từ checkpoint, hoặc bắt đầu giai đoạn học tiếp với lịch mới.
- Sinh văn bản bằng greedy, temperature, top-k và top-p; có demo terminal và trình duyệt.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Chạy các lệnh từ thư mục gốc của repo.

```powershell
git clone https://github.com/cuong897/mini-gpt.git
cd mini-gpt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m minigpt env
```

Trên Linux/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`. Nếu PowerShell không cho activate, dùng `.\.venv\Scripts\python.exe` thay cho `python`.

`requirements.txt` khai báo khoảng phiên bản hỗ trợ. `requirements-lock.txt` ghi phiên bản các thư viện trực tiếp đã dùng, không phải lock đầy đủ mọi dependency. Muốn chạy CUDA cần bản PyTorch phù hợp với môi trường GPU; dùng `--device cpu` để chạy CPU hoặc `--device auto` để tự chọn.

## Chạy thử offline

```powershell
python scripts/smoke.py --device cpu --root runs/smoke_quickstart
```

Lệnh này tạo dữ liệu giả, học tokenizer, huấn luyện cả ba mô hình nhỏ, đánh giá, sinh mẫu và xuất bảng cùng biểu đồ vào `runs/smoke_quickstart/comparison`.

Chọn một `--root` mới nếu đã chạy trước đó. Smoke kiểm tra luồng hoạt động; kết quả trên dữ liệu giả không đại diện cho chất lượng tiếng Việt thực tế.

## Chuẩn bị dữ liệu thật

Tải một tập nhỏ từ Wikipedia tiếng Việt rồi chuẩn bị tokenizer và các tập train/validation/test:

```powershell
python -m minigpt fetch --documents 300 --buffer-size 300 --output data/raw/wiki_300.jsonl
python -m minigpt prepare --source data/raw/wiki_300.jsonl --output data/wiki_300 --vocab-size 8000 --ratios 0.8 0.1 0.1 --kind real
python scripts/pilot.py --data data/wiki_300 --root runs/pilot_quickstart --device auto
```

Bước tải cần Internet. Metadata nguồn ghi revision và SHA256; có thể truyền `fetch --revision SHA` để cố định revision cho lần tải sau. Pilot chỉ dùng ngân sách huấn luyện nhỏ để kiểm tra tích hợp.

Có thể dùng JSONL UTF-8 riêng với trường `text` bắt buộc và các trường nguồn như `id`, `url`, `title`:

```json
{"id":"article-1","url":"https://example.com/article-1","title":"Tiêu đề","text":"Nội dung tài liệu tiếng Việt..."}
```

Đọc `manifest.json` sau khi chuẩn bị để kiểm tra số tài liệu, token, tỷ lệ chia và fingerprint. Giữ thông tin nguồn và tuân thủ điều kiện sử dụng của dữ liệu.

## Huấn luyện

Ví dụ dưới đây dùng tập `data/wiki_300` đã tạo ở trên. Với thí nghiệm chính, hãy chuẩn bị corpus phù hợp và thay đường dẫn `--data`.

```powershell
python -m minigpt ngram --data data/wiki_300 --run runs/main/ngram --context 256 --orders 2 3 4 --alphas 1 10 100 --kind main
python -m minigpt train --config configs/lstm_main.json --data data/wiki_300 --run runs/main/lstm --device auto
python -m minigpt train --config configs/gpt_main.json --data data/wiki_300 --run runs/main/gpt --device auto
```

Các cấu hình trong `configs/` gồm mức `smoke`, `pilot`, `main` và các kích thước GPT. `gpt_main.json` dùng context 256, hidden dimension 384, 6 lớp, 6 attention head, batch size 8 và gradient accumulation 4; ngân sách là 20 triệu target token. Số tham số thực phụ thuộc kích thước vocabulary.

N-gram đếm toàn bộ tập train, còn các mô hình neural chạy theo ngân sách cấu hình. Khi so sánh, cần ghi rõ dữ liệu và lượng token đã xử lý của từng mô hình. Nhãn `main` không tự chứng minh chất lượng đầu ra.

### Tiếp tục một lượt huấn luyện

```powershell
python -m minigpt train --config configs/gpt_main.json --data data/wiki_300 --run runs/main/gpt --device auto --resume runs/main/gpt/last.pt
```

`last.pt` dùng để khôi phục trạng thái huấn luyện trong cùng thư mục run, với cấu hình và fingerprint dữ liệu phù hợp. `best.pt` được chọn theo validation và dùng mặc định cho đánh giá, sinh văn bản. Có thể dùng `--stop-after 100` để dừng có checkpoint trước khi chạy tiếp.

`--continue-from` dành cho một giai đoạn học mới: giữ trạng thái optimizer/RNG từ checkpoint nguồn, đặt lại lịch và bộ đếm của giai đoạn mới. Không dùng đồng thời với `--resume`.

## Đánh giá và sinh văn bản

```powershell
python -m minigpt evaluate --data data/wiki_300 --run runs/main/gpt --split test --device auto
python -m minigpt generate --data data/wiki_300 --run runs/main/gpt --prompt "Việt Nam là một quốc gia" --max-new-tokens 100 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 42 --output runs/main/gpt/sample.json
```

Đổi `--run` sang thư mục N-gram hoặc LSTM tương ứng để dùng baseline. Thêm `--greedy` nếu muốn chọn token có xác suất lớn nhất. Perplexity thấp hơn không bảo đảm văn bản mạch lạc; nên kiểm tra cả mẫu sinh và mức lặp.

Xuất bảng và biểu đồ từ các lượt đã đánh giá:

```powershell
python scripts/summarize.py --runs runs/main/ngram runs/main/lstm runs/main/gpt --output runs/main/comparison
```

Script kiểm tra tính tương thích của dữ liệu, tokenizer, context và giao thức trước khi gộp kết quả.

## Demo

Demo terminal dùng đường dẫn dữ liệu và checkpoint do bạn chọn:

```powershell
python scripts/demo.py --data data/wiki_300 --run runs/main/gpt --device auto
```

Demo trình duyệt:

```powershell
python -m minigpt.demo_web --device auto --port 7860
```

Mở `http://127.0.0.1:7860`. Hiện demo web dùng các đường dẫn cố định trong `model_paths()` tại `minigpt/demo_web.py`: dữ liệu `data/wiki_reviewed_v1` với baseline/pilot ở `runs/reviewed_pilot`, và GPT ở `runs/continuation_20261105/gpt` với dữ liệu `data/wiki_main_20261102_v2`. Cần có các artifact tương ứng hoặc sửa hàm này để trỏ tới run của bạn. Bản clone mới chưa có checkpoint nên chưa thể sinh văn bản qua các preset này.

## Kiểm thử

```powershell
python -m pytest -q --basetemp=.cache/pytest-local
```

Bộ test kiểm tra chia dữ liệu, tokenizer, kiến trúc, generation, tính nhất quán artifact, resume và demo. Các test HTTP cần quyền mở socket loopback.

## Cấu trúc mã nguồn

| Đường dẫn | Nội dung |
| --- | --- |
| `minigpt/data.py`, `minigpt/cleaning.py` | Chuẩn hóa, loại trùng, chia dữ liệu và BPE |
| `minigpt/models.py` | LSTM, causal attention và GPT |
| `minigpt/ngram.py` | N-gram và bộ đếm SQLite |
| `minigpt/engine.py` | Huấn luyện, checkpoint, resume và đánh giá |
| `minigpt/generation.py` | Sinh văn bản và sampling |
| `minigpt/experiments.py` | Kiểm tra tính nhất quán kết quả thí nghiệm |
| `minigpt/demo_web.py`, `web/demo/` | API local và giao diện demo |
| `configs/` | Cấu hình mô hình và ngân sách huấn luyện |
| `scripts/` | Chạy thử, so sánh và chẩn đoán thí nghiệm |
| `tests/` | Kiểm thử tự động |

Dữ liệu, checkpoint, cache và kết quả chạy được tạo trên máy; không đi kèm bản clone. Repo không công bố sẵn một bộ trọng số hay điểm benchmark có thể tải xuống. Với mô hình nhỏ huấn luyện từ đầu, đầu ra có thể lặp, sai thông tin hoặc thiếu mạch lạc.
