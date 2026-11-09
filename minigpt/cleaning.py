"""Conservative prose extraction; no rewriting or factual correction."""
import html
import re
from collections import Counter

from .data import normalize

TAILS = {'tham khảo', 'chú thích', 'liên kết ngoài', 'tài liệu tham khảo', 'đọc thêm', 'nguồn tham khảo'}
ACCENTED = re.compile('[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', re.I)


def remove_nested(text, opening, closing):
    """Drop balanced markup; retain an unmatched opener for later rejection."""
    result, i = [], 0
    while i < len(text):
        if not text.startswith(opening, i):
            result.append(text[i]); i += 1
            continue
        start, depth = i, 1
        i += len(opening)
        while i < len(text) and depth:
            if text.startswith(opening, i):
                depth += 1; i += len(opening)
            elif text.startswith(closing, i):
                depth -= 1; i += len(closing)
            else:
                i += 1
        if depth:
            result.append(text[start:])
        else:
            result.append(' ')
    return ''.join(result)


def clean_prose(text):
    counts = Counter()
    text = normalize(text)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
    text = re.sub(r'<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref\s*>', ' ', text, flags=re.I | re.S)
    text = re.sub(r'<(?:table|script|style)\b[^>]*>.*?</(?:table|script|style)\s*>', ' ', text, flags=re.I | re.S)
    text = remove_nested(text, '{{', '}}')
    text = remove_nested(text, '{|', '|}')
    text = re.sub(r'\[\[(?:File|Image|Tập tin|Hình|Category|Thể loại):[^\]]*\]\]', ' ', text, flags=re.I)
    text = re.sub(r'\[\[([^\[\]]+)\]\]', lambda m: m[1].split('|')[-1], text)
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'<(?:br|/?p)\b[^>]*>', '\n\n', text, flags=re.I)
    text = re.sub(r'</?[A-Za-z][^>]*>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r"'{2,5}", '', text)
    lines = []
    for line in text.splitlines():
        heading = line.strip().strip('=').strip().casefold()
        if heading in TAILS:
            counts['reference_tail_removed'] += 1
            break
        if (re.match(r'^\s*[*#;|!]', line) or re.fullmatch(r'\s*https?://\S+\s*', line)
                or (line.strip() and len(line.split()) < 10 and not re.search(r'[.!?…]\s*$', line))):
            counts['list_or_heading_lines_removed'] += 1
            lines.append('')
        else:
            lines.append(line)
    paragraphs, seen = [], set()
    for paragraph in normalize('\n'.join(lines)).split('\n\n'):
        paragraph = ' '.join(paragraph.split())
        letters = sum(c.isalpha() for c in paragraph)
        if (len(paragraph) < 160 or len(paragraph.split()) < 25
                or letters / max(1, len(paragraph)) < .60
                or len(ACCENTED.findall(paragraph)) / max(1, letters) < .03
                or not re.search(r'[.!?…]', paragraph)
                or any(marker in paragraph for marker in ['{{', '}}', '[[', ']]', '\ufffd'])
                or re.search(r'https?://|</?[A-Za-z]', paragraph)):
            counts['non_prose_paragraphs_removed'] += 1
            continue
        key = paragraph.casefold()
        if key in seen:
            counts['duplicate_paragraphs_removed'] += 1
            continue
        seen.add(key)
        paragraphs.append(paragraph)
    return '\n\n'.join(paragraphs), counts
