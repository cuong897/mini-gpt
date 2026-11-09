const $ = (id) => document.getElementById(id);
let catalog, running = false, lastResult = null;
const number = (value) => value == null ? '—' : value.toLocaleString('vi-VN');
const preset = () => catalog.presets.find((p) => p.id === $('preset').value);

function text(tag, value, className) {
  const el = document.createElement(tag);
  el.textContent = value;
  if (className) el.className = className;
  return el;
}
function cards(results = null) {
  $('results').replaceChildren();
  const list = results || preset().models.map((info) => ({ id: info.id, info }));
  for (const item of list) {
    const info = item.info;
    const card = text('article', '', 'card');
    const heading = text('h3', info.label);
    heading.append(text('span', info.kind === 'main' ? 'TRAIN CHÍNH' : 'PILOT', `badge ${info.kind === 'main' ? 'main' : ''}`));
    card.append(heading);
    card.append(text('div', `${number(info.documents)} bài · vocab ${number(info.vocabulary)}\n${number(info.training_tokens)} target train${info.parameters ? ` · ${number(info.parameters)} tham số` : ''}`, 'meta'));
    if (item.error || !info.available) {
      card.append(text('p', item.error || info.reason, 'text error'));
    } else if (item.result) {
      const output = text('div', '', 'text');
      output.append(text('span', item.result.prompt, 'prefix'), document.createTextNode(item.result.continuation));
      card.append(output);
      const reason = item.result.stop_reason === 'eos' ? 'gặp EOS' : 'đủ giới hạn token';
      card.append(text('div', `${item.result.generated_ids.length} token mới · ${item.seconds.toFixed(2)} giây · Dừng: ${reason}`, 'result-footer'));
    } else {
      card.append(text('p', 'Văn bản sẽ xuất hiện ở đây.', 'empty'));
    }
    $('results').append(card);
  }
}
function selection() {
  if (running) return;
  lastResult = null;
  $('download').hidden = true;
  $('choices').replaceChildren(text('legend', 'Mô hình sẽ viết tiếp'));
  for (const info of preset().models) {
    const label = text('label', '', `choice${info.available ? '' : ' disabled'}`);
    const input = document.createElement('input');
    input.type = 'checkbox'; input.value = info.id;
    input.checked = info.available; input.disabled = !info.available;
    label.append(input, text('span', info.label), text('small', info.kind === 'main' ? 'chính' : 'pilot'));
    $('choices').append(label);
  }
  $('notice').textContent = preset().id === 'latest'
    ? 'Mini-GPT dùng bản 29,4 triệu tham số đã học tổng cộng 40 triệu token, được chọn theo validation. N-gram và LSTM dùng bản pilot 300 bài; GPT dùng dữ liệu 10.000 bài (9.999 bài sau loại trùng). Khác dữ liệu, tokenizer và ngân sách train: dùng để minh họa, không xếp hạng chất lượng trực tiếp.'
    : 'Cả ba dùng cùng dữ liệu và tokenizer pilot 300 bài. N-gram đếm toàn bộ train; LSTM/GPT mới học 16.384 target token. Cùng dữ liệu chưa có nghĩa là cùng ngân sách huấn luyện.';
  $('status').textContent = 'Chọn mô hình và nhấn “Sinh văn bản”.';
  $('submit').disabled = !preset().models.some((m) => m.available);
  cards();
}
function busy(value) {
  running = value;
  $('submit').disabled = value;
  $('preset').disabled = value;
  $('results').classList.toggle('loading', value);
  // Freeze the prompt/settings too, so the displayed request matches its result.
  for (const el of $('form').querySelectorAll('input,textarea,button')) {
    if (el.id !== 'submit') el.disabled = value;
  }
  if (!value) {
    for (const input of $('choices').querySelectorAll('input')) {
      input.disabled = !preset().models.find((m) => m.id === input.value).available;
    }
  }
}
$('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (running) return;
  const models = [...$('choices').querySelectorAll('input:checked')].map((el) => el.value);
  if (!models.length) { $('status').textContent = 'Hãy chọn ít nhất một mô hình.'; return; }
  const request = { preset: $('preset').value, models, prompt: $('prompt').value };
  for (const id of ['max_new_tokens', 'seed', 'temperature', 'top_k', 'top_p']) request[id] = Number($(id).value);
  busy(true); lastResult = null; $('download').hidden = true;
  $('results').replaceChildren(text('p', 'Đang sinh văn bản…', 'empty'));
  $('status').textContent = 'Đang chạy lần lượt các mô hình. Lần đầu có thể cần vài giây để nạp checkpoint…';
  try {
    const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Không thể sinh văn bản');
    lastResult = result; cards(result.results); $('download').hidden = false;
    const errors = result.results.filter((r) => r.error).length;
    $('status').textContent = errors ? `Hoàn tất, có ${errors} mô hình chưa sinh được. Xem thông tin trong thẻ.` : 'Đã sinh xong. Bạn có thể đổi đoạn mở đầu và thử lại.';
  } catch (error) {
    $('results').replaceChildren(text('p', error.message, 'error'));
    $('status').textContent = 'Chưa có kết quả mới. Kiểm tra kết nối hoặc thông số rồi thử lại.';
  } finally { busy(false); }
});
$('preset').addEventListener('change', selection);
for (const button of document.querySelectorAll('[data-prompt]')) button.addEventListener('click', () => { $('prompt').value = button.dataset.prompt; $('prompt').focus(); });
$('download').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = `mini-gpt-demo-${Date.now()}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
fetch('/api/models').then((r) => { if (!r.ok) throw new Error('Không đọc được danh sách mô hình'); return r.json(); }).then((value) => {
  catalog = value;
  for (const p of catalog.presets) { const option = text('option', p.label); option.value = p.id; $('preset').append(option); }
  $('preset').value = catalog.default_preset; selection();
}).catch((error) => { $('status').textContent = error.message; });
