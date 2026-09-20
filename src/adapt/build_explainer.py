"""Build a self-contained HTML walkthrough from actual demo artifacts; no API calls."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'results/gate_demo/langfuse_20260906')
    parser.add_argument('--output', type=Path, default=ROOT/'_planner/demo.html')
    args = parser.parse_args()
    records = json.loads((args.run/'cases.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    audit_path = args.run/'trace_audit.json'
    payload = dict(cases=records, manifest=manifest,
                   audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None)
    previous = ROOT/'results/gate_demo/live_20260906/cases.jsonl'
    payload['previous'] = [json.loads(line) for line in previous.read_text(encoding='utf-8').splitlines()] if previous.exists() else []
    template = Path(__file__).with_name('explainer.template.html').read_text(encoding='utf-8')
    # Escape HTML delimiters so an injected context cannot break out of the JSON script.
    data = json.dumps(payload, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('__DEMO_DATA__', data), encoding='utf-8')
    print(f'Built {args.output}: {len(records)} cases, embedded data, no external assets.')


if __name__ == '__main__':
    main()
