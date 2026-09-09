"""Tracing dùng chung cho các thí nghiệm, dùng Langfuse Python SDK (v4).

Mỗi bước bọc trong `step()`: tạo một observation lồng nhau qua
`langfuse.start_as_current_observation`, log input/output/metadata, tự đóng khi thoát
khối `with`. Thiếu `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` thì tự động chỉ log ra
stdout, không gọi Langfuse — pipeline chạy y như cũ.

`type` truyền vào phải là một observation type hợp lệ của Langfuse: "span",
"generation", "tool", "retriever", "agent", "embedding", "evaluator", "event",
"guardrail". Xem https://langfuse.com/docs/observability/features/observation-types.

Khác `adapt_tracing.py` ở gốc repo (dùng Braintrust, gắn với EhrAgent) ở backend và
tên project, nhưng cùng giao diện `step()`/`Step.log()`/`flush()` để chỗ gọi
(`agent.py`, `run_demo.py`) không cần biết bên dưới là backend nào.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

logger = logging.getLogger("toolpoison.trace")

_client = None
_state = None  # None = chưa thử, True/False = đã thử


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _init() -> bool:
    global _client, _state
    if _state is not None:
        return _state
    _state = False
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse tắt (thiếu LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY) - chỉ log ra stdout.")
        return _state
    try:
        from langfuse import get_client

        _client = get_client()
        _state = True
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        logger.info("Langfuse bật, host=%s", host)
    except Exception as e:  # pragma: no cover - phụ thuộc môi trường
        logger.warning("Không bật được Langfuse (%s) - chỉ log ra stdout.", e)
    return _state


def _max_chars() -> int:
    try:
        return int(os.environ.get("LANGFUSE_MAX_CHARS", "8000"))
    except ValueError:
        return 8000


def _sanitize(value, _depth: int = 0):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        limit = _max_chars()
        return value if len(value) <= limit else value[:limit] + f"... [+{len(value) - limit} chars]"
    if _depth >= 4:
        return _sanitize(str(value), _depth)
    if isinstance(value, dict):
        return {str(k): _sanitize(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v, _depth + 1) for v in value]
    return _sanitize(str(value), _depth)


class Step:
    def __init__(self, name: str, observation) -> None:
        self.name = name
        self.observation = observation

    def trace_link(self) -> dict:
        """Correlation only: an allocated trace ID does not confirm remote delivery."""
        trace_id = getattr(self.observation, "trace_id", None)
        url = None
        if trace_id and _client is not None:
            try:
                url = _client.get_trace_url(trace_id=trace_id)
            except Exception:
                pass
        return {"trace_id": trace_id, "trace_url": url}

    def log(self, **fields) -> None:
        if self.observation is None:
            return
        try:
            self.observation.update(**{k: _sanitize(v) for k, v in fields.items()})
        except Exception as e:  # pragma: no cover
            logger.debug("update lỗi ở bước %s: %s", self.name, e)

    def set_output(self, output) -> None:
        self.log(output=output)


@contextmanager
def step(name: str, type: str = "span", input=None, metadata=None, model: str | None = None):
    started = time.time()
    logger.info("[start] %s", name)
    observation = None
    ctx = None
    if _init():
        try:
            kwargs: dict = {"as_type": type, "name": name}
            if input is not None:
                kwargs["input"] = _sanitize(input)
            if metadata:
                kwargs["metadata"] = _sanitize(metadata)
            if model is not None:
                kwargs["model"] = model
            ctx = _client.start_as_current_observation(**kwargs)
            observation = ctx.__enter__()
        except Exception as e:  # pragma: no cover
            logger.debug("Không tạo được observation %s: %s", name, e)
            observation = None
            ctx = None
    handle = Step(name, observation)
    try:
        yield handle
    except Exception as e:
        handle.log(level="ERROR", status_message=repr(e))
        logger.exception("[fail]  %s sau %.2fs", name, time.time() - started)
        if ctx is not None:
            try:
                ctx.__exit__(e.__class__, e, None)
            except Exception:
                pass
            ctx = None
        raise
    else:
        logger.info("[done]  %s (%.2fs)", name, time.time() - started)
    finally:
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass


def flush() -> None:
    """Đẩy event còn trong hàng đợi. Lưu ý: Langfuse v4 export nền, `flush()` KHÔNG đảm bảo
    gửi xong trước khi process thoát — script ngắn phải gọi `shutdown()` ở cuối."""
    if _init():
        try:
            _client.flush()
        except Exception as e:  # pragma: no cover
            logger.debug("flush lỗi: %s", e)


def shutdown() -> None:
    """Flush + shutdown, block tới khi mọi span được gửi. Gọi ở cuối script trước khi thoát.

    `flush()` một mình không đủ: Langfuse v4 dựa trên OpenTelemetry BatchSpanProcessor gửi
    span ở luồng nền; nếu process thoát ngay sau `flush()` thì span cuối có thể chưa kịp POST.
    `shutdown()` chặn cho tới khi exporter gửi hết (khuyến nghị của Langfuse cho script ngắn).
    """
    if _init():
        try:
            _client.flush()
        except Exception as e:  # pragma: no cover
            logger.debug("flush lỗi: %s", e)
        try:
            _client.shutdown()
        except Exception as e:  # pragma: no cover
            logger.debug("shutdown lỗi: %s", e)
