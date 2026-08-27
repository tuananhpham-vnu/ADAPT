"""In danh sách target của Makefile (các dòng bắt đầu bằng `## name: mô tả`)."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

path = sys.argv[1] if len(sys.argv) > 1 else "Makefile"
with open(path, encoding="utf-8") as f:
    for line in f:
        if line.startswith("## ") and ":" in line:
            name, _, desc = line[3:].partition(":")
            print(f"  {name.strip():<16} {desc.strip()}")
