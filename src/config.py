"""Cấu hình đọc từ biến môi trường cho lớp `src/providers`.

`src/providers/*` import `get_settings()` từ đây. File này trước đó bị thiếu nên
cả package không import được.

Ba vai model của pipeline ARTEMIS (xem `_guidance/10_artemis_overview.md`) được
cấu hình riêng biệt:

- internal: phân rã system prompt, sinh test case và criteria.
- test:     chạy agent đang được kiểm thử.
- judge:    chấm response theo criteria.

Tách riêng vì paper yêu cầu judge model khác test model để tránh bias khi judge
ưu ái response do chính nó sinh ra (Sect. 7.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:  # python-dotenv là tuỳ chọn; thiếu thì đọc thẳng os.environ
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    """Đọc biến môi trường, bỏ khoảng trắng và dấu nháy thừa.

    `.env` trong repo có nhiều giá trị được bọc nháy kép, ví dụ
    DEEPSEEK_MODEL_NAME="deepseek-chat".
    """
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'")
    return value or default


@dataclass(frozen=True)
class Settings:
    # --- OpenAI và mọi endpoint tương thích OpenAI ---
    openai_model_name: str
    openai_base_url: str

    # --- DeepSeek ---
    deepseek_base_url: str
    deepseek_model_name: str

    # --- Gemini ---
    gemini_model_name: str

    # --- Anthropic ---
    anthropic_model_name: str

    # --- OpenRouter ---
    openrouter_base_url: str
    openrouter_model_name: str

    # --- Ba vai model của pipeline ---
    internal_provider: str
    internal_model: str
    test_provider: str
    test_model: str
    judge_provider: str
    judge_model: str

    # --- Tham số chạy ---
    # n_run: số lần chạy lại mỗi test case. Cần > 1 để đo được instability.
    n_run: int
    # n_judge: số vòng chấm mỗi response. RQ4 của paper cho thấy 1 là đủ khi
    # judge chạy ở temperature 0 (chênh accuracy tối đa 0.01 giữa 1 và 5 vòng),
    # trong khi bước judge chiếm phần lớn chi phí token.
    n_judge: int
    temperature: float

    def role(self, name: str) -> tuple[str, str]:
        """Trả về (provider, model) cho vai `internal`, `test` hoặc `judge`."""
        try:
            return {
                "internal": (self.internal_provider, self.internal_model),
                "test": (self.test_provider, self.test_model),
                "judge": (self.judge_provider, self.judge_model),
            }[name]
        except KeyError:
            raise ValueError(
                f"Vai không hợp lệ: {name!r}. Chọn internal, test hoặc judge."
            ) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    deepseek_model = _env("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    gemini_model = _env("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")

    return Settings(
        openai_model_name=_env("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        deepseek_base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model_name=deepseek_model,
        gemini_model_name=gemini_model,
        anthropic_model_name=_env("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-5"),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model_name=_env("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini"),
        internal_provider=_env("ARTEMIS_INTERNAL_PROVIDER", "gemini"),
        internal_model=_env("ARTEMIS_INTERNAL_MODEL", gemini_model),
        test_provider=_env("ARTEMIS_TEST_PROVIDER", "deepseek"),
        test_model=_env("ARTEMIS_TEST_MODEL", deepseek_model),
        judge_provider=_env("ARTEMIS_JUDGE_PROVIDER", "gemini"),
        judge_model=_env("ARTEMIS_JUDGE_MODEL", gemini_model),
        n_run=int(_env("ARTEMIS_N_RUN", "5")),
        n_judge=int(_env("ARTEMIS_N_JUDGE", "1")),
        temperature=float(_env("ARTEMIS_TEMPERATURE", "0")),
    )


def check_role_separation(settings: Settings | None = None) -> str | None:
    """Cảnh báo nếu judge và test dùng cùng một model.

    Paper dùng judge khác test model làm biện pháp construct validity: judge có xu
    hướng cho điểm cao hơn với response do chính model đó sinh ra. Trả về chuỗi
    cảnh báo, hoặc None nếu hai vai đã tách.
    """
    settings = settings or get_settings()
    if (settings.judge_provider, settings.judge_model) == (
        settings.test_provider,
        settings.test_model,
    ):
        return (
            f"judge và test đang dùng chung {settings.judge_provider}/{settings.judge_model}. "
            "Điểm accuracy sẽ bị lệch do self-preference bias; đặt ARTEMIS_JUDGE_PROVIDER "
            "hoặc ARTEMIS_JUDGE_MODEL khác đi."
        )
    return None
