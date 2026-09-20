"""Step-level tracing cho pipeline EhrAgent.

Mỗi bước được bọc trong `step("<ten_buoc>")`:
  - in log ra stdout (theo dõi khi chạy terminal)
  - tạo một span cùng tên trên Braintrust (khi có BRAINTRUST_API_KEY)

Thiếu package `braintrust` hoặc thiếu API key thì mọi thứ tự động degrade về
log thường, pipeline chạy y như cũ.

Env:
  BRAINTRUST_API_KEY    bật tracing (bắt buộc)
  BRAINTRUST_PROJECT    tên project trên Braintrust (mặc định: ADAPT-EhrAgent)
  BRAINTRUST_MAX_CHARS  cắt chuỗi dài trước khi gửi (mặc định 8000)
"""

import logging
import os
import time
from contextlib import contextmanager

logger = logging.getLogger("ehragent.trace")

_bt = None            # module braintrust, nếu import được
_bt_logger = None     # đối tượng trả về từ init_logger
_state = None         # None = chưa thử, True/False = đã thử


def setup_logging(level=logging.INFO):
    """Bật log dạng `HH:MM:SS | ehragent.trace | ...` cho toàn bộ pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _init():
    """Khởi tạo Braintrust logger một lần duy nhất (lazy)."""
    global _bt, _bt_logger, _state
    if _state is not None:
        return _state
    _state = False
    if not os.environ.get("BRAINTRUST_API_KEY"):
        logger.info("Braintrust tắt (thiếu BRAINTRUST_API_KEY) - chỉ log ra stdout.")
        return _state
    try:
        import braintrust

        _bt = braintrust
        _bt_logger = braintrust.init_logger(
            project=os.environ.get("BRAINTRUST_PROJECT", "ADAPT-EhrAgent")
        )
        _state = True
        logger.info(
            "Braintrust bật, project=%s",
            os.environ.get("BRAINTRUST_PROJECT", "ADAPT-EhrAgent"),
        )
    except Exception as e:  # pragma: no cover - phụ thuộc môi trường
        logger.warning("Không bật được Braintrust (%s) - chỉ log ra stdout.", e)
    return _state


def is_enabled():
    return _init()


def _max_chars():
    try:
        return int(os.environ.get("BRAINTRUST_MAX_CHARS", "8000"))
    except ValueError:
        return 8000


def _sanitize(value, _depth=0):
    """Ép về kiểu JSON-friendly và cắt bớt chuỗi quá dài."""
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
    if hasattr(value, "tolist"):  # numpy / torch
        try:
            return _sanitize(value.tolist(), _depth + 1)
        except Exception:
            pass
    return _sanitize(str(value), _depth)


class Step:
    """Handle của một bước: dùng để log input/output/metadata bổ sung."""

    def __init__(self, name, span):
        self.name = name
        self.span = span

    def log(self, **fields):
        if self.span is None:
            return
        try:
            self.span.log(**{k: _sanitize(v) for k, v in fields.items()})
        except Exception as e:  # pragma: no cover
            logger.debug("span.log lỗi ở bước %s: %s", self.name, e)

    def set_output(self, output):
        self.log(output=output)

    def set_metadata(self, **metadata):
        self.log(metadata=metadata)


@contextmanager
def step(name, type="function", input=None, metadata=None):
    """Bọc một bước pipeline thành span tên `name` trên Braintrust.

    type: "task" | "function" | "llm" | "tool" | "score" (chuẩn Braintrust).
    """
    started = time.time()
    logger.info("[start] %s", name)
    span = None
    if _init():
        try:
            kwargs = {"name": name, "type": type}
            if input is not None:
                kwargs["input"] = _sanitize(input)
            if metadata:
                kwargs["metadata"] = _sanitize(metadata)
            span = _bt.start_span(**kwargs)
            span.__enter__()
        except Exception as e:  # pragma: no cover
            logger.debug("Không tạo được span %s: %s", name, e)
            span = None
    handle = Step(name, span)
    try:
        yield handle
    except Exception as e:
        handle.log(error=repr(e))
        logger.exception("[fail]  %s sau %.2fs", name, time.time() - started)
        if span is not None:
            try:
                span.__exit__(type(e), e, None)
            except Exception:
                pass
            span = None
        raise
    else:
        logger.info("[done]  %s (%.2fs)", name, time.time() - started)
    finally:
        if span is not None:
            try:
                span.__exit__(None, None, None)
            except Exception:
                pass


def wrap_openai_client(client):
    """Bọc client OpenAI để mỗi lần gọi API tự sinh span llm (prompt/usage/latency)."""
    if not _init():
        return client
    try:
        return _bt.wrap_openai(client)
    except Exception as e:  # pragma: no cover
        logger.debug("wrap_openai lỗi: %s", e)
        return client


def flush():
    """Đẩy nốt span còn trong buffer (gọi trước khi thoát chương trình)."""
    if _init():
        try:
            _bt_logger.flush()
        except Exception as e:  # pragma: no cover
            logger.debug("flush lỗi: %s", e)
