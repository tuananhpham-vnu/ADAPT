"""Vietnamese Markdown walkthrough, generated entirely from saved experiment logs.

Rebuild an existing run without calling the LLM:
python -m src.agentpoison.attack_report --run results/agentpoison/<timestamp>
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def cell(value):
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', '<br>').replace('\r', '').replace('`', '&#96;')


def fence(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    delimiter = '`' * max(3, 1 + max((len(m[0]) for m in re.finditer(r'`+', text)), default=0))
    return f'{delimiter}text\n{text}\n{delimiter}'


def fraction(n, d):
    return f'{n}/{d} ({n/d:.0%})' if d else 'Chưa có dữ liệu'


def condition(poisoned, triggered):
    return ('Memory có mẫu độc' if poisoned else 'Memory sạch') + ' · ' + ('Có trigger' if triggered else 'Không trigger')


def call_text(call):
    args = ', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in call.get('args', {}).items())
    return f"{call.get('name', '?')}({args})"


def write_report(output: Path, final: dict) -> Path:
    manifest, records = final['manifest'], final['records']
    ok = [r for r in records if r.get('status') == 'ok']
    expected = manifest.get('n_base_queries', 0) * manifest.get('n_conditions', 4)
    status = 'Hoàn tất' if final.get('status') == 'completed' else 'Chưa hoàn tất / có lỗi'
    mode = manifest.get('poison_mode', 'unknown')
    lines = ['# Báo cáo demo AgentPoison', '',
        f"**Trạng thái:** {status}. Đã ghi {len(records)}/{expected} lượt; {len(ok)} lượt thành công, {len(records)-len(ok)} lượt lỗi.", '',
        '## 1. Lần chạy này làm gì?', '',
        f"- Model: **{cell(manifest.get('provider', '?'))} / {cell(manifest.get('model', 'chưa nạp'))}**.",
        f"- Retriever: **{cell(manifest.get('retriever', '?'))}**, lấy top-{manifest.get('top_k', '?')} theo độ tương đồng embedding.",
        f"- {manifest.get('n_base_queries', '?')} câu hỏi gốc × 4 điều kiện; cùng câu hỏi chỉ đổi memory và việc thêm trigger.",
        f"- Memory sạch: {manifest.get('clean_memory_size', '?')} mẫu. Memory có độc: {manifest.get('poisoned_memory_size', '?')} mẫu tổng cộng.",
        f"- Trigger: “{cell(manifest.get('trigger', ''))}”. Đây là chuỗi có sẵn, **không được tối ưu lại trong lệnh demo**.",
        '- Các tool chỉ thực thi giả lập. Không có mutation, repair hoặc gate trong demo này.', '',
        '**Cách đưa mẫu độc vào prompt:** ' + (
            '`demo` — chỉ đưa cặp query mẫu → tool-call mẫu vào few-shot; không chèn thêm note như chỉ thị.' if mode == 'demo' else
            '`instruction` — ngoài few-shot còn đưa note truy hồi vào prompt dưới dạng policy; đây là biến thể có chỉ thị bổ sung.' if mode == 'instruction' else cell(mode)), '',
        '```text\nQuery → DPR tìm top-k ví dụ → ghép prompt → LLM chọn tool/tham số → thực thi giả lập\n```', '',
        'Retriever chọn **ví dụ** chứ không bỏ phiếu chọn “tool chung”. LLM có thể đọc mẫu độc nhưng vẫn chọn tool đúng.', '',
        '## 2. Kết quả bốn điều kiện', '',
        'Các tỷ lệ chỉ tính trên lượt chạy thành công của từng điều kiện. Lượt lỗi không được tính là tấn công thất bại.', '',
        '| Điều kiện | Thành công / đã chạy | Lỗi | Truy hồi trúng mẫu độc | Tra cứu đúng | Chọn chuyển tới đích độc | Chuyển tới đích độc khi đã truy hồi trúng |',
        '|---|---:|---:|---:|---:|---:|---:|']
    groups = {}
    for poisoned in (False, True):
        for triggered in (False, True):
            rows = [r for r in records if r['poisoned_memory'] == poisoned and r['triggered'] == triggered]
            good = [r for r in rows if r.get('status') == 'ok']
            groups[poisoned, triggered] = good
            hit = [r for r in good if r.get('poisoned_hit')]
            lines.append(f"| {condition(poisoned, triggered)} | {len(good)}/{len(rows)} | {len(rows)-len(good)} | "
                f"{fraction(len(hit), len(good))} | {fraction(sum(bool(r.get('correct_lookup')) for r in good),len(good))} | "
                f"{fraction(sum(bool(r.get('target_transfer')) for r in good),len(good))} | "
                f"{fraction(sum(bool(r.get('target_transfer')) for r in hit),len(hit))} |")
    attack = groups[True, True]
    untriggered = groups[True, False]
    hits = [r for r in attack if r.get('poisoned_hit')]
    targets = [r for r in attack if r.get('target_transfer')]
    retained = [r for r in hits if r.get('correct_lookup')]
    leakage = [r for r in untriggered if r.get('poisoned_hit')]
    lines += ['', '**Cách đọc:** “truy hồi trúng mẫu độc” đo tầng retrieval; “chọn chuyển tới đích độc” đo hành vi model '
              '(gọi transfer_funds tới đích attacker đã cấy). Hai tỷ lệ không đồng nghĩa. Cột điều kiện cuối dùng số lượt truy hồi trúng làm mẫu số.', '',
        '## 3. Insight từ lần chạy này', '',
        f"- **Tầng retrieval:** trong memory có độc, query có trigger kéo mẫu độc vào top-k ở {fraction(len(hits),len(attack))}; không trigger là {fraction(len(leakage),len(untriggered))}.",
        f"- **Tầng hành động:** với memory có độc và query có trigger, có {fraction(len(targets),len(attack))} lượt chọn chuyển tới đích độc.",
        f"- **Sau khi mẫu độc đã lọt top-k:** {fraction(len(retained),len(hits))} lượt vẫn tra cứu đúng. Đây là bằng chứng cần tách retrieval khỏi hành vi."]
    if hits and not targets:
        lines += ['- **Kết luận quan sát:** truy hồi đã lấy được mẫu độc, nhưng lần chạy này chưa làm model chuyển tới đích đối kháng. Không nên gọi đó là tấn công hành động thành công.']
    elif targets:
        lines += ['- **Kết luận quan sát:** có proposal chuyển tới đích đối kháng; mở từng case bên dưới để kiểm tra mẫu đã truy hồi và tool-call cụ thể.']
    if leakage:
        lines += ['- **Trigger chưa tách biệt hoàn toàn:** mẫu độc vẫn được truy hồi ở một số query không có trigger: ' + ', '.join(cell(r['case_id']) for r in leakage) + '.']
    lines += ['- **Giới hạn:** memory nhỏ, số mẫu giữa hai nhóm khác nhau, mỗi điều kiện/query chỉ chạy một lần. Chưa tái lập thuật toán tối ưu trigger hoặc chứng minh kết quả tổng quát của paper.', '',
              '## 4. Tra nhanh từng case', '',
              '| Case | Điều kiện | Có mẫu độc trong top-k? | Model gọi gì? | Kết quả |',
              '|---|---|---|---|---|']
    for i,r in enumerate(records,1):
        succeeded = r.get('status') == 'ok'
        outcome = 'Lỗi chạy' if not succeeded else 'Chuyển tới đích độc' if r.get('target_transfer') else 'Tra cứu đúng' if r.get('correct_lookup') else 'Hành vi khác / sai tác vụ'
        lines.append(f"| [{cell(r['case_id'])}](#case-{i}) | {condition(r['poisoned_memory'],r['triggered'])} | "
                     f"{('Có' if r.get('poisoned_hit') else 'Không') if succeeded else 'Không kết luận'} | "
                     f"{cell('; '.join(call_text(c) for c in r.get('tool_calls',[])) or 'Không có lời gọi được ghi')} | {outcome} |")
    lines += ['', '## 5. Giải thích từng case', '', 'Các case giữ nguyên thứ tự thực chạy; bốn điều kiện được xáo trộn theo seed đã lưu.', '']
    for i,r in enumerate(records,1):
        steps = {s['name']:s for s in r.get('steps',[])}
        lines += [f'<a id="case-{i}"></a>', '', f"### {i}. {cell(r['case_id'])}", '',
                  f"**Điều kiện:** {condition(r['poisoned_memory'],r['triggered'])}.", '',
                  '**Bước 1 — Câu hỏi gửi agent**', '', fence(r['query']), '',
                  'Yêu cầu gốc là tra cứu số dư đúng tài khoản, không cho phép chuyển tiền.', '']
        if r.get('status') != 'ok':
            lines += [f"**Lỗi chạy:** {cell(r.get('error_type','không xác định'))}. Không đủ log để kết luận retrieval hay hành động thành công.", '', '---', '']
            continue
        docs = steps.get('retrieve-demos',{}).get('output',[])
        lines += ['**Bước 2 — Retriever lấy những mẫu nào?**', '',
                  '| Hạng | Query mẫu trong memory | Điểm cosine | Loại mẫu | Tool của mẫu |',
                  '|---:|---|---:|---|---|']
        for rank,d in enumerate(docs,1):
            arguments = d.get('tool_args')
            if arguments is None:
                matches = [m for m in manifest.get('memory',[]) if m.get('query') == d.get('query') and m.get('tool_name') == d.get('tool')]
                if len(matches)==1: arguments=matches[0].get('tool_args')
            call = call_text({'name':d.get('tool','?'),'args':arguments or {}})
            lines.append(f"| {rank} | {cell(d.get('query',''))} | {cell(d.get('score','—'))} | {'**ĐỘC**' if d.get('poisoned') else 'Sạch'} | {cell(call)} |")
        if not docs: lines += ['| — | Không có dữ liệu top-k trong log | — | — | — |']
        lines += ['', 'Điểm cosine là độ tương đồng embedding, không phải xác suất model sẽ làm theo mẫu.', '',
                  '**Bước 3 — LLM đã đọc prompt nào?**', '',
                  ('Các ví dụ top-k được ghép vào system prompt dưới dạng few-shot.' if mode=='demo' else
                   'Các ví dụ top-k và note/policy được ghép vào system prompt.'), '',
                  '<details>', '<summary>Mở prompt đầy đủ đã gửi LLM</summary>', '']
        for message in steps.get('build-prompt',{}).get('output',[]):
            lines += [f"**Role: {cell(message.get('role','?'))}**", '', fence(message.get('content','')), '']
        lines += ['</details>', '', '**Bước 4 — Model chọn tool và thực thi giả lập**', '']
        calls = r.get('tool_calls',[])
        lines += [fence('\n'.join(call_text(c) for c in calls) or 'Không có tool-call.'), '']
        for result in r.get('results') or []:
            if isinstance(result,dict):
                lines += ['- Kết quả giả lập: ' + '; '.join(f'**{cell(k)}** = {cell(v)}' for k,v in result.items()) + '.']
            else: lines += ['- Kết quả giả lập: ' + cell(result)]
        if not r.get('results'): lines += ['Không có kết quả thực thi được ghi.']
        generation = steps.get('select-tool',{})
        text = generation.get('output',{}).get('text')
        if text: lines += ['', '<details><summary>Văn bản model trả thêm</summary>', '', fence(text), '', '</details>']
        usage = generation.get('usage') or {}
        lines += ['', f"Token: vào **{usage.get('input','chưa ghi')}**, ra **{usage.get('output','chưa ghi')}**.", '',
                  '**Bước 5 — Insight của case**', '', cell(r.get('insight','Chưa có insight trong log.')), '']
        if r.get('poisoned_hit') and r.get('correct_lookup') and not r.get('target_transfer'):
            lines += ['Mẫu độc đã lọt vào prompt, nhưng model vẫn thực hiện đúng ý định tra cứu. Tầng retrieval bị tác động chưa đủ để lật hành động.', '']
        if r.get('poisoned_hit') and not r['triggered']:
            lines += ['Mẫu độc xuất hiện dù không có trigger: lưu ý ảnh hưởng ngoài điều kiện kích hoạt mong muốn.', '']
        url = r.get('langfuse',{}).get('trace_url')
        if url and urlsplit(url).scheme == 'https':
            lines += [f"[Mở trace Langfuse](<{url.replace('>', '%3E').replace('<', '%3C')}>) — cần quyền project; link không tự chứng minh dữ liệu đã gửi thành công.", '']
        lines += ['---', '']
    lines += ['## 6. File để đối chiếu', '',
              '- [final.json](final.json): toàn bộ cấu hình, thống kê và records.',
              '- [summary.json](summary.json): thống kê bốn điều kiện.',
              '- [records.json](records.json): dữ liệu chi tiết từng case.',
              '- [records.jsonl](records.jsonl): log theo thứ tự chạy.',
              '- [manifest.json](manifest.json): cấu hình và memory gốc.', '']
    output = Path(output)
    path = output/'REPORT.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    final=json.loads((args.run/'final.json').read_text(encoding='utf-8'))
    print(write_report(args.run,final))


if __name__=='__main__':
    main()
