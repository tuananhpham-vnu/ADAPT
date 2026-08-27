"""In ra trigger cuối cùng của lần tối ưu gần nhất.

    python scripts/show_trigger.py [--agent qa] [--algo ap] [--save_dir ./results]
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--agent", "-a", default="qa")
parser.add_argument("--algo", "-al", default="ap")
parser.add_argument("--save_dir", "-s", default="./results")
args = parser.parse_args()

pattern = os.path.join(args.save_dir, args.agent, args.algo, "*", "stdout.txt")
runs = sorted(glob.glob(pattern), key=os.path.getmtime)
if not runs:
    raise SystemExit(f"Không tìm thấy run nào khớp {pattern}")

path = runs[-1]
last = None
iters = 0
with open(path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("Iteration:"):
            iters += 1
        if "Current adv_passage" in line:
            last = line.strip()

print(f"run:        {path}")
print(f"iterations: {iters}")
print(last or "chưa có dòng 'Current adv_passage' nào — run có thể còn đang chạy hoặc đã lỗi")
