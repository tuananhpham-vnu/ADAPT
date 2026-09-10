# _planner

Kế hoạch gốc duy nhất: [PLAN.md](PLAN.md).

Hướng dẫn chạy AgentPoison trên SSH/Linux bằng các lệnh Python trực tiếp, gồm optimize trigger thật,
chạy từng case, resume và đọc artifact: [RUN_SSH.md](RUN_SSH.md).

Mở [demo.html](demo.html) bằng trình duyệt để xem demo đang chạy gì, khám phá từng bước
của 10 case, đọc insight và so sánh hướng phát triển với AgentPoison. HTML nhúng dữ liệu,
không cần server hay API key; link Langfuse cần quyền project.

Tạo lại HTML từ log:

```powershell
.venv-adapt/Scripts/python.exe src/toolpoison/build_explainer.py
```

Gồm hướng nghiên cứu, lộ trình, backlog ý tưởng và demo 10 case đã chạy có log/insight.
