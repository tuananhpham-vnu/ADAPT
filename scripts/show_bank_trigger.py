"""In ra trigger cuối cùng, và quá trình tìm ra nó, từ một bank của stage `bank`.

Đọc các artifact do `python -m src.triggers.hierarchy compare` sinh ra
(<bank>/<position>/<arm>.json), không phải log của algo/trigger_optimization.py.

    python scripts/show_bank_trigger.py outputs/specificity/kaggle/bank
    python scripts/show_bank_trigger.py outputs/specificity/kaggle/bank --history
    python scripts/show_bank_trigger.py outputs/specificity/kaggle/bank --arm per_query --group 3 --history
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("bank", type=Path, help="thư mục bank (chứa prefix/, suffix/, ...)")
parser.add_argument("--position", "-p", default=None, help="mặc định: mọi position có trong bank")
parser.add_argument("--arm", "-a", default=None, help="mặc định: mọi arm")
parser.add_argument("--group", "-g", type=int, default=None, help="chỉ một nhóm/trigger")
parser.add_argument("--history", action="store_true", help="in cả đường đi của search")
parser.add_argument("--improved-only", action="store_true",
                    help="với --history, chỉ in các bước cải thiện best feasible")
args = parser.parse_args()

positions = [args.position] if args.position else \
    sorted(d.name for d in args.bank.iterdir() if d.is_dir())
if not positions:
    raise SystemExit(f"không có position nào dưới {args.bank}")

for position in positions:
    pdir = args.bank / position
    arms = [args.arm] if args.arm else sorted(f.stem for f in pdir.glob("*.json"))
    for arm in arms:
        path = pdir / f"{arm}.json"
        if not path.exists():
            print(f"-- bỏ qua {path} (không tồn tại)")
            continue
        art = json.loads(path.read_text(encoding="utf-8"))
        bank = art["bank"]
        fits = bank["training"]
        print(f"\n=== {position} / {arm} "
              f"({len(bank['triggers'])} trigger, routing={bank['routing']}, "
              f"budget {bank['triggers'] and fits[0]['budget_used']}"
              f"/{fits[0]['budget_limit'] if fits else '?'} mỗi nhóm) ===")
        for i, (trigger, fit) in enumerate(zip(bank["triggers"], fits)):
            if args.group is not None and i != args.group:
                continue
            flag = "" if fit["constraint_feasible"] else "   [constraint_feasible=false]"
            print(f"  [{i}] {trigger!r}")
            print(f"      loss={fit['loss']:.4f}  sim={fit['semantic_similarity']:.4f}  "
                  f"origin={fit['origin']}  evaluations={fit['evaluations']}{flag}")
            if not args.history:
                continue
            steps = fit.get("history", [])
            if args.improved_only:
                steps = [s for s in steps if s.get("improved_feasible_best")]
            print(f"      -- {len(steps)} bước --")
            for s in steps:
                mark = "*" if s.get("improved_feasible_best") else " "
                ok = "" if s["valid"] else "  (meaning guard fail)"
                print(f"      {mark} budget={s['budget_used']:>5}  loss={s['loss']:+.4f}  "
                      f"{s['origin']:<9} {s['trigger']!r}{ok}")

        for split in ("validation", "test"):
            m = art.get("evaluations", {}).get(split, {}).get("metrics")
            if m:
                cells = " ".join(f"{k}={m[k]:.3f}" for k in
                                 ("retrieval_rate", "meaning_pass_rate", "joint_success_rate",
                                  "false_activation", "worst_group_joint_success") if k in m)
                print(f"  {split:<11} n={m.get('n', '?')} {cells}")
print()
